"""TrackingModel — 4SM-based predictor for the v0 dataset.

Composition: front-end (conv_stem | patch_embed) → StackedSpikingSSM → residual head.

The model predicts in **normalised** coordinates: the image is mapped to
`[-1, 1] × [-1, 1]`. This makes gradients well-conditioned (no need to
learn a constant ~64 px output bias) and matches the head's zero-init.

The output is **residual on top of the per-bin event centroid** (the user's
fix #2b): for each prediction step we compute the centroid of events in the
input window in pixel space, normalise it, and add a learned correction:

    pred_norm = centroid_norm(bins) + delta_norm(SSM(bins))

`delta_norm` is what the head outputs and starts near zero, so the model's
day-zero prediction is just the centroid — which already gets ~19 px val
RMSE without any learning. The SSM only has to learn the trajectory-
extrapolation correction, which is a much better-posed task than
"detect-then-predict" from scratch.

Front-ends
----------
- `conv_stem`: 3× strided conv on each binned frame → AdaptiveAvgPool → linear
  → one D-vector per bin. SSM sequence length = T (= 200 for v0).

- `patch_embed`: ViT-style patch embed per frame → P tokens per bin (with
  intra-frame positional embedding). Patches are flattened across time, so
  the SSM sequence length = T·P. Per-bin predictions come from mean-pooling
  the SSM outputs of that bin's P tokens.

The selective-SSM core is `StackedSpikingSSM` from src/main_spikingjelly_selective_c.py.
"""
from __future__ import annotations

import os
import sys
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

# ssm_module.py sits beside this file in the release package.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ssm_module import StackedSpikingSSM  # noqa: E402


# ----- Front-ends ---------------------------------------------------------


class ConvStem(nn.Module):
    """One token per bin. (B, T, H, W, C) → (B, T, D).

    Three strided convs reduce the input down, then pool to a SMALL spatial
    map (default 4×4 = 16 cells), flatten and linear-project to D. Keeping a
    16-cell map (instead of the old 1×1 global pool) preserves coarse spatial
    information per bin: the linear projection can learn to up-weight cells
    containing the moving target relative to noisy cells. The old global-
    pool collapsed everything into a single "how many events fired" scalar
    per channel, which under heavy DVS noise destroys the signal we need.
    """

    def __init__(self, in_channels: int = 2, d_model: int = 128,
                 hidden_channels: tuple[int, int, int] = (32, 64, 128),
                 pool_size: int = 4):
        super().__init__()
        c1, c2, c3 = hidden_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(c1, c2, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(pool_size),    # (B*T, c3, pool, pool)
        )
        self.pool_size = pool_size
        # Spatial features land flat: pool² · c3. Linear-project to d_model.
        self.proj = nn.Linear(c3 * pool_size * pool_size, d_model)

    @property
    def tokens_per_bin(self) -> int:
        return 1

    def forward(self, bins: torch.Tensor) -> torch.Tensor:
        # bins: (B, T, H, W, C)
        B, T, H, W, C = bins.shape
        x = bins.permute(0, 1, 4, 2, 3).reshape(B * T, C, H, W)
        x = self.stem(x).flatten(1)        # (B*T, c3 · pool²)
        x = self.proj(x)                    # (B*T, D)
        return x.view(B, T, -1)             # (B, T, D)


class PatchEmbed(nn.Module):
    """ViT-style patch tokenization. (B, T, H, W, C) → (B, T·P, D).

    Each frame becomes P = (H/patch_size) · (W/patch_size) tokens with a
    learnable per-patch positional embedding. Patches from one frame are
    laid out contiguously in the sequence, then frames are concatenated in
    chronological order.
    """

    def __init__(self, in_channels: int = 2, d_model: int = 128,
                 patch_size: int = 32, image_hw: tuple[int, int] = (128, 128)):
        super().__init__()
        H, W = image_hw
        if H % patch_size or W % patch_size:
            raise ValueError(f"image {image_hw} not divisible by patch_size={patch_size}")
        self.patch_size = patch_size
        self.num_patches = (H // patch_size) * (W // patch_size)
        # Conv2d with kernel = stride = patch_size implements patch embed.
        self.proj = nn.Conv2d(in_channels, d_model,
                              kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    @property
    def tokens_per_bin(self) -> int:
        return self.num_patches

    def forward(self, bins: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = bins.shape
        x = bins.permute(0, 1, 4, 2, 3).reshape(B * T, C, H, W)
        x = self.proj(x)                                  # (B*T, D, h, w)
        x = x.flatten(2).transpose(1, 2)                   # (B*T, P, D)
        x = x + self.pos_embed                             # add positional
        return x.view(B, T * self.num_patches, -1)         # (B, T·P, D)


# ----- Full model --------------------------------------------------------


def compute_centroids_window(
    bins: torch.Tensor,
    window_bins: int = 3,
) -> torch.Tensor:
    """Per-bin centroid of events in the trailing `window_bins`-bin window.

    bins: `(B, T, H, W, C)` with non-negative event counts (ON, OFF channels
    are summed). Returns `(B, T, 2)` in **pixel** coordinates. When the
    window is empty (no events), the centroid falls back to the image
    centre — that's the only reasonable prior.
    """
    B, T, H, W, C = bins.shape
    counts = bins.sum(dim=-1)  # (B, T, H, W) total events per bin per pixel

    # Sliding sum over the last `window_bins` bins (causal). We left-pad the
    # time axis so bin t still gets a valid window even at the start.
    if window_bins > 1:
        # F.pad pads dimensions starting from the last; we want to pad time
        # which is dim=1 in (B, T, H, W). Easiest: do an explicit cumsum
        # diff trick.
        cumulative = counts.cumsum(dim=1)  # (B, T, H, W)
        # window from bin t-window_bins+1 to t inclusive:
        #   sum = cumulative[t] - cumulative[t-window_bins]   (with 0 for negative idx)
        zeros = counts.new_zeros((B, window_bins, H, W))
        padded = torch.cat([zeros, cumulative], dim=1)  # (B, T+window_bins, H, W)
        windowed = cumulative - padded[:, : T]
    else:
        windowed = counts

    total = windowed.sum(dim=(-1, -2))  # (B, T)
    # Per-pixel coordinate grids (cached on first call by torch op cache).
    ys = torch.arange(H, dtype=windowed.dtype, device=windowed.device)
    xs = torch.arange(W, dtype=windowed.dtype, device=windowed.device)
    sum_per_row = windowed.sum(dim=-1)   # (B, T, H)  — sum of pixels with the same y
    sum_per_col = windowed.sum(dim=-2)   # (B, T, W)  — sum of pixels with the same x
    cy = (sum_per_row * ys).sum(dim=-1)
    cx = (sum_per_col * xs).sum(dim=-1)

    safe_total = total.clamp(min=1.0)
    cy = cy / safe_total
    cx = cx / safe_total

    # Empty window → fall back to image centre.
    empty = total < 1e-6
    if empty.any():
        cx = torch.where(empty, torch.full_like(cx, W * 0.5), cx)
        cy = torch.where(empty, torch.full_like(cy, H * 0.5), cy)

    return torch.stack([cx, cy], dim=-1)  # (B, T, 2)


class TrackingModel(nn.Module):
    """Front-end → StackedSpikingSSM → residual head.

    Predictions live in normalised coordinates `[-1, 1]^2`:

        pred_norm = centroid_norm(bins) + delta_norm

    `centroid_norm` is the centroid of events in the 30 ms input window
    mapped from pixel space to `[-1, 1]`. `delta_norm` is what the SSM head
    outputs (≈0 at init). Convert back to pixels with
    `pix = (pred_norm + 1) * (image_hw / 2)` (or the helpers below).

    Forward input:  bins      (B, T, H, W, C)  with C = 2 (ON, OFF counts)
    Forward output: pred_norm (B, T, 2)         in [-1, 1]^2
    """

    def __init__(
        self,
        frontend: Literal["conv_stem", "patch_embed"] = "conv_stem",
        *,
        # SSM core
        n_layers: int = 2,
        d_model: int = 128,
        state_dim: int = 256,
        conv_kernel: int = 3,            # 3 bins = 30 ms input window per spec
        dropout: float = 0.1,
        use_z_gate: bool = False,
        ssm_kwargs: dict | None = None,
        # Front-end
        in_channels: int = 2,
        image_hw: tuple[int, int] = (128, 128),
        patch_size: int = 32,
        conv_pool_size: int = 4,         # ConvStem spatial pool — 1 = global (v0), 4 = 16-cell map (v1)
        # Head
        head_hidden: int = 128,
        n_horizons: int = 1,             # v5: heads per timestep. 1 = v0-v4 single-horizon.
        # Centroid window (in 10 ms bins). 3 = match the 30 ms input window.
        centroid_window_bins: int = 3,
    ):
        super().__init__()

        ssm_kwargs = ssm_kwargs or {}
        self.image_hw = tuple(image_hw)
        self.centroid_window_bins = int(centroid_window_bins)
        self.conv_pool_size = int(conv_pool_size)
        self.n_horizons = int(n_horizons)

        if frontend == "conv_stem":
            self.frontend = ConvStem(in_channels=in_channels, d_model=d_model,
                                     pool_size=conv_pool_size)
        elif frontend == "patch_embed":
            self.frontend = PatchEmbed(
                in_channels=in_channels, d_model=d_model,
                patch_size=patch_size, image_hw=image_hw,
            )
        else:
            raise ValueError(f"unknown frontend {frontend!r}")
        self.frontend_name = frontend

        self.core = StackedSpikingSSM(
            n_layers=n_layers,
            d_model=d_model,
            state_dim=state_dim,
            conv_kernel=conv_kernel,
            dropout=dropout,
            use_z_gate=use_z_gate,
            **ssm_kwargs,
        )
        # Override the core's input projection: it expects scalar pixels
        # (`Linear(1, d_model)`), but our front-end already produces
        # d_model-dim tokens. Replace with identity.
        self.core.input_proj = nn.Identity()

        # Residual head: outputs one (x, y) per horizon, in normalised coords.
        # Day-zero prediction = centroid for every horizon (bias zeroed below).
        # Output shape pre-reshape: (B, T, n_horizons * 2).
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, self.n_horizons * 2),
        )
        with torch.no_grad():
            # Zero the bias so every horizon's initial prediction is the centroid.
            self.head[-1].bias.zero_()

    # ----- coordinate helpers (used by training / evaluation) -------------

    def pix_to_norm(self, xy_pix: torch.Tensor) -> torch.Tensor:
        """(B,T,2) pixel coords → [-1, 1] normalised coords."""
        H, W = self.image_hw
        scale = xy_pix.new_tensor([W * 0.5, H * 0.5])
        return xy_pix / scale - 1.0

    def norm_to_pix(self, xy_norm: torch.Tensor) -> torch.Tensor:
        """(B,T,2) [-1, 1] normalised coords → pixel coords."""
        H, W = self.image_hw
        scale = xy_norm.new_tensor([W * 0.5, H * 0.5])
        return (xy_norm + 1.0) * scale

    def reset_state(self, B: int, device=None):
        self.core.reset_state(B, device)

    def forward(self, bins: torch.Tensor) -> torch.Tensor:
        """Predictions in normalised coords ∈ [-1, 1]^2.

        Output shape:
          - `n_horizons == 1`: (B, T, 2) — back-compat with v0-v4 inference.
          - `n_horizons > 1`:  (B, T, H, 2) — one prediction per horizon.

        The centroid baseline is broadcast across all horizons; each horizon
        gets its own residual delta from the head.
        """
        # 1. Centroid baseline (pixel → normalised). No grad through this.
        with torch.no_grad():
            centroid_pix = compute_centroids_window(bins, self.centroid_window_bins)
            centroid_norm = self.pix_to_norm(centroid_pix)              # (B, T, 2)

        # 2. Front-end: (B, T, H, W, C) → (B, T_eff, D)
        tokens = self.frontend(bins)
        B, T_eff, D = tokens.shape

        # 3. SSM expects (T, B, D)
        tokens = tokens.transpose(0, 1).contiguous()
        feats = self.core.forward_vectorized(tokens)       # (T_eff, B, D)

        # 4. Pool back to one prediction per bin if needed
        tpb = self.frontend.tokens_per_bin
        if tpb == 1:
            per_bin = feats                                 # (T, B, D)
        else:
            T = T_eff // tpb
            per_bin = feats.view(T, tpb, B, D).mean(dim=1)  # (T, B, D)

        # 5. Multi-horizon residual head.
        T_steps = per_bin.shape[0]
        delta_norm = self.head(per_bin)                     # (T, B, n_horizons*2)
        delta_norm = delta_norm.transpose(0, 1).contiguous()  # (B, T, n_horizons*2)

        if self.n_horizons == 1:
            # Back-compat single-horizon output: (B, T, 2)
            return centroid_norm + delta_norm
        # Multi-horizon: reshape and broadcast centroid across horizons
        delta_norm = delta_norm.view(B, T_steps, self.n_horizons, 2)
        pred = centroid_norm.unsqueeze(2) + delta_norm       # (B, T, H, 2)
        return pred


# ----- Backward-compatible checkpoint loading ---------------------------


def _infer_conv_pool_size(state_dict: dict, c3: int = 128) -> int | None:
    """Read frontend.proj.weight shape and back out the pool_size used at
    training time. Returns None for non-conv_stem checkpoints.

    proj.weight is (d_model, c3 · pool²). Given d_model and c3 we can solve
    for pool: pool = sqrt(in_features / c3).
    """
    w = state_dict.get("frontend.proj.weight")
    if w is None:
        return None
    in_features = w.shape[1]
    pool_squared = in_features // c3
    if pool_squared <= 0 or pool_squared * c3 != in_features:
        return None
    pool = int(pool_squared ** 0.5)
    if pool * pool != pool_squared:
        return None
    return pool


def build_from_args(args_dict: dict,
                    state_dict: dict | None = None) -> "TrackingModel":
    """Construct a TrackingModel from a checkpoint's saved args + state_dict.

    Handles backward compatibility: detects `conv_pool_size` from the
    saved state_dict shape and `use_z_gate` from the presence of z_gate
    keys, when the CLI args don't record them (true for every checkpoint
    saved before each respective CLI flag was added — v0 used pool=1
    no z_gate, v1+ use pool=4, v4 uses pool=8 z_gate=True).
    """
    image_hw = tuple(args_dict.get("image_hw") or (128, 128))
    pool_size = args_dict.get("conv_pool_size")
    if pool_size is None and state_dict is not None:
        inferred = _infer_conv_pool_size(state_dict)
        if inferred is not None:
            pool_size = inferred
    if pool_size is None:
        pool_size = 4  # current default

    use_z_gate = args_dict.get("use_z_gate")
    if use_z_gate is None and state_dict is not None:
        use_z_gate = any("z_gate" in k for k in state_dict.keys())
    if use_z_gate is None:
        use_z_gate = False

    # v5: n_horizons baked into head[-1].weight shape (out_features = n_horizons * 2).
    n_horizons = args_dict.get("n_horizons")
    if n_horizons is None and state_dict is not None:
        # Last Linear of the head: index 3 of the Sequential.
        w = state_dict.get("head.3.weight")
        if w is not None and w.shape[0] % 2 == 0:
            n_horizons = w.shape[0] // 2
    if n_horizons is None:
        n_horizons = 1

    return TrackingModel(
        frontend=args_dict["frontend"],
        n_layers=args_dict.get("n_layers", 2),
        d_model=args_dict.get("d_model", 128),
        state_dim=args_dict.get("state_dim", 256),
        conv_kernel=args_dict.get("conv_kernel", 3),
        dropout=args_dict.get("dropout", 0.1),
        patch_size=args_dict.get("patch_size", 32),
        image_hw=image_hw,
        conv_pool_size=pool_size,
        use_z_gate=use_z_gate,
        n_horizons=n_horizons,
    )


# ----- Smoke test -------------------------------------------------------


def _smoke():
    """Quick forward pass to verify shapes and the residual centroid output."""
    B, T, H, W, C = 2, 200, 128, 128, 2
    bins = torch.zeros(B, T, H, W, C)
    # Stamp a single ON event at a known position in each bin so the
    # centroid is well-defined and not the fallback image centre.
    bins[:, :, 64, 80, 0] = 1.0  # one ON event at (x=80, y=64) per bin

    for fe in ("conv_stem", "patch_embed"):
        print(f"\n--- frontend={fe} ---")
        m = TrackingModel(frontend=fe, n_layers=2, d_model=64, state_dim=128,
                          patch_size=32)
        n_params = sum(p.numel() for p in m.parameters())
        print(f"  params: {n_params/1e6:.3f} M  tokens_per_bin: {m.frontend.tokens_per_bin}")
        m.reset_state(B)
        m.eval()
        with torch.no_grad():
            out_norm = m(bins)
            out_pix = m.norm_to_pix(out_norm)
        print(f"  out shape: {tuple(out_norm.shape)}  (expect ({B}, {T}, 2))")
        print(f"  out range (norm): [{out_norm.min().item():.3f}, {out_norm.max().item():.3f}]")
        # With residual-from-centroid + ~zero init, prediction at bin 100
        # should be close to (80, 64) in pixel space.
        pix_at_100 = out_pix[0, 100].tolist()
        print(f"  example pred[B=0, t=100] in pixels: ({pix_at_100[0]:.2f}, {pix_at_100[1]:.2f})  "
              f"(expect ≈ (80.00, 64.00))")
        assert out_norm.shape == (B, T, 2), f"bad shape {out_norm.shape}"


if __name__ == "__main__":
    _smoke()
