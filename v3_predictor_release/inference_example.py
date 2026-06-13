"""End-to-end inference example for the v3 predictor.

Loads best.pt, generates a synthetic event sequence with the included
Python event simulator (a ball moving down the image), bins it into
10-ms windows, and runs the model. Prints the predicted (cx, cy) for
each bin and the GT pixel position for sanity.

This standalone script is the simplest possible "did the model load
correctly" smoke test for the evaluation team.

Run:
    python inference_example.py
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import torch

# Make ssm_module + model_loader importable when running from the
# package root.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from event_sim import EventSim
from model_loader import build_from_args


CHECKPOINT_PATH = os.path.join(_HERE, "best.pt")


def render_ball_frame(image_hw: tuple[int, int], cx: float, cy: float,
                      radius_px: int = 6) -> np.ndarray:
    """Render a single grayscale frame with a ball at (cx, cy).

    Same shift-aware sub-pixel rendering used by the synthetic training
    pipeline: cv2.circle with LINE_AA and shift=4 for sub-pixel accuracy.
    """
    H, W = image_hw
    frame = np.zeros((H, W), dtype=np.uint8)
    shift = 4
    factor = 1 << shift
    cv2.circle(
        frame,
        (int(round(cx * factor)), int(round(cy * factor))),
        radius_px * factor,
        color=150,             # gray; ESIM cares about log-intensity gradient
        thickness=-1,
        lineType=cv2.LINE_AA,
        shift=shift,
    )
    return frame


def bin_events(events: np.ndarray, n_bins: int, bin_s: float,
               image_hw: tuple[int, int]) -> np.ndarray:
    """events (N, 4) [x, y, t_s, polarity] -> (T, H, W, 2) float32.

    Channel 0 = ON event count per pixel in the bin.
    Channel 1 = OFF event count per pixel in the bin.
    """
    H, W = image_hw
    bins = np.zeros((n_bins, H, W, 2), dtype=np.float32)
    if events.size == 0:
        return bins
    t_idx = np.floor(events[:, 2] / bin_s).astype(np.int64)
    keep = (t_idx >= 0) & (t_idx < n_bins)
    events = events[keep]
    t_idx = t_idx[keep]
    x = events[:, 0].astype(np.int64).clip(0, W - 1)
    y = events[:, 1].astype(np.int64).clip(0, H - 1)
    on = events[:, 3] > 0
    np.add.at(bins, (t_idx[on], y[on], x[on], 0), 1.0)
    np.add.at(bins, (t_idx[~on], y[~on], x[~on], 1), 1.0)
    return bins


def main():
    # 1) Load the v3 checkpoint.
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    print(f"  epoch = {ckpt.get('epoch')}, val_rmse_px = {ckpt.get('val_rmse_px'):.3f}")
    print(f"  trained on image_hw = {ckpt['args']['image_hw']}")

    # 2) Build the network from the checkpoint's `args` block and load weights.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_from_args(ckpt["args"], state_dict=ckpt["model_state_dict"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  device = {device}")

    # 3) Simulate a 2-second clip at 1 kHz: ball moves vertically (cx fixed,
    # cy goes from 80 -> 180 -> 80 over the clip). Match the training pipeline.
    image_hw = tuple(ckpt["args"]["image_hw"])     # (260, 260)
    H, W = image_hw
    frame_hz = 1000.0
    duration_s = 2.0
    n_frames = int(round(duration_s * frame_hz))

    cx_path = np.full(n_frames, W / 2)              # column 130
    cy_path = 130 + 50 * np.sin(2 * np.pi * 0.5 * np.arange(n_frames) / frame_hz)

    sim = EventSim(image_hw, cp=0.2, cn=0.2, refractory_s=1e-4,
                   log_eps=1e-3, use_log=True)
    f0 = render_ball_frame(image_hw, cx_path[0], cy_path[0])
    sim.reset(f0, t=0.0)

    chunks = []
    for i in range(1, n_frames):
        t = i / frame_hz
        f = render_ball_frame(image_hw, cx_path[i], cy_path[i])
        ev = sim.step(f, t)
        if ev.size:
            chunks.append(ev)
    events = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 4), dtype=np.float32)
    print(f"  generated {events.shape[0]} events over {duration_s:.1f} s")

    # 4) Bin events into 10-ms windows (the model's expected input).
    bin_ms = 10.0
    n_bins = int(round(duration_s * 1000.0 / bin_ms))
    bins = bin_events(events, n_bins, bin_ms * 1e-3, image_hw)
    print(f"  binned -> {bins.shape}  total {int(bins.sum())} events kept")

    # 5) Forward pass through the model.
    bins_t = torch.from_numpy(bins).unsqueeze(0).to(device)    # (1, T, H, W, 2)
    if hasattr(model, "reset_state"):
        model.reset_state(1, device=device)
    with torch.no_grad():
        pred_norm = model(bins_t)
        # Multi-horizon checkpoints (not v3) return (B, T, H, 2); v3 returns (B, T, 2).
        if pred_norm.dim() == 4:
            horizons = ckpt["args"].get("horizons_ms", [50.0])
            idx = horizons.index(50.0) if 50.0 in horizons else 0
            pred_norm = pred_norm[:, :, idx, :]
        pred_pix = model.norm_to_pix(pred_norm).squeeze(0).cpu().numpy()  # (T, 2)

    # 6) Compare to GT. The predictor outputs the ball position at
    # t + 50 ms relative to the end of the bin.
    horizon_ms = 50.0
    target_t_ms = (np.arange(n_bins) + 1) * bin_ms + horizon_ms
    target_idx = np.clip(np.round(target_t_ms * frame_hz / 1000.0).astype(int) - 1,
                         0, n_frames - 1)
    gt_x = cx_path[target_idx]
    gt_y = cy_path[target_idx]

    valid = slice(3, n_bins)       # skip the warm-up bins
    diff_x = pred_pix[valid, 0] - gt_x[valid]
    diff_y = pred_pix[valid, 1] - gt_y[valid]
    rmse = float(np.sqrt(np.mean(diff_x ** 2 + diff_y ** 2)))
    print(f"\nResult (post-warmup): {valid.stop - valid.start} bins")
    print(f"  RMSE pred vs GT projection:  {rmse:.2f} px")
    print(f"  pred x range: [{pred_pix[valid, 0].min():.1f}, {pred_pix[valid, 0].max():.1f}]")
    print(f"  pred y range: [{pred_pix[valid, 1].min():.1f}, {pred_pix[valid, 1].max():.1f}]")
    print(f"  GT y range:   [{gt_y[valid].min():.1f}, {gt_y[valid].max():.1f}]")

    print("\nFirst 10 (pred_cx, pred_cy) vs (gt_cx, gt_cy) at end-of-bin + 50 ms:")
    print(f"{'tick':>4}  {'pred_cx':>8}  {'pred_cy':>8}  {'gt_cx':>8}  {'gt_cy':>8}")
    for t in range(3, 13):
        print(f"{t:>4}  {pred_pix[t, 0]:>8.2f}  {pred_pix[t, 1]:>8.2f}  "
              f"{gt_x[t]:>8.2f}  {gt_y[t]:>8.2f}")


if __name__ == "__main__":
    main()
