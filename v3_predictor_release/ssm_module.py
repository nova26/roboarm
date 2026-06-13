#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pMNIST training script for a Mamba-style *selective* diagonal SSM with spiking gates.

High-level model/CLI documentation lives in `README.md` (kept in sync with this file).
For a selective-SSM vs Mamba checklist see `docs/MAMBA_SELECTIVE_COMPARISON.md`.
"""
from __future__ import annotations

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import math
import random
import time
import sys
import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from tqdm.auto import tqdm
from spikingjelly.activation_based import surrogate as sj_surrogate
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for servers
import matplotlib.pyplot as plt

# -----------------------------
# Dependencies: SpikingJelly + TensorFlow
# -----------------------------
try:
    from spikingjelly.activation_based import neuron, functional
except ImportError as e:
    raise SystemExit("Please install SpikingJelly first:  pip install spikingjelly\n" + str(e))

try:
    import tensorflow as tf
except ImportError as e:
    raise SystemExit(
        "This script builds pMNIST from tf.keras.datasets to match your baseline.\n"
        "Install TensorFlow (CPU is fine):  pip install tensorflow-cpu\n" + str(e)
    )

try:
    import torch._inductor.config as ic
    ic.max_autotune_gemm = False   # turn off the heavy GEMM autotuner
    # optional: also reduce autotune globally
    ic.max_autotune = False
except Exception:
    pass


# -----------------------------
# Small helpers needed by class initializers
# -----------------------------
def softplus_inv(y: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    y = torch.clamp(y, min=eps)
    return y + torch.log1p(-torch.exp(-y))

DEFAULT_SEED: int = 7

# -----------------------------
# Model blocks
# -----------------------------
class PMNISTCached(Dataset):
    def __init__(self, cache_file: str, expected_perm_seed: int | None = None):
        obj = torch.load(cache_file, map_location="cpu")
        self.x: torch.Tensor = obj["x"].contiguous()  # [N,784,1] in [0,1]
        self.y: torch.Tensor = obj["y"].contiguous()
        self.perm_seed: int = int(obj.get("perm_seed", -1))
        self.is_sequential: bool = bool(obj.get("is_sequential", False))

        # Validate cache matches expected permutation seed
        if expected_perm_seed is not None and self.perm_seed != expected_perm_seed:
            raise ValueError(
                f"Cache file perm_seed mismatch: {cache_file} has perm_seed={self.perm_seed}, "
                f"but expected {expected_perm_seed}. Cache may be corrupted or mislabeled."
            )
        # Additional safety: for sMNIST (perm_seed < 0), require the cache to be marked sequential.
        if expected_perm_seed is not None and int(expected_perm_seed) < 0 and not self.is_sequential:
            raise ValueError(
                f"Cache file is not marked sequential (is_sequential={self.is_sequential}) but "
                f"perm_seed={expected_perm_seed} indicates sMNIST. Rebuild cache in {Path(cache_file).parent}."
            )
    def __len__(self): return self.x.shape[0]
    def __getitem__(self, idx: int): return self.x[idx], self.y[idx]

class SpikingReadout(nn.Module):
    """Spiking MLP -> analog output via EMA (synapse)."""
    def __init__(self, in_dim, hidden, out_dim, tau=2.0, v_threshold=1.0, v_reset=0.0, syn_alpha=0.8):
        super().__init__()
        self.enc = nn.Linear(in_dim, hidden, bias=True)
        # SpikingJelly LIF neuron
        self.lif = neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, 
                                  surrogate_function=sj_surrogate.Sigmoid(alpha=2.0))
        self.dec = nn.Linear(hidden, out_dim, bias=True)
        self.syn_alpha = syn_alpha
        self.syn_state = None
        self.register_buffer("_dummy", torch.empty(0))
        
    def reset(self, batch_size, device=None):
        dev = device or next(self.parameters()).device
        # Reset SpikingJelly neuron state
        functional.reset_net(self.lif)
        self.syn_state = torch.zeros(batch_size, self.dec.out_features, device=dev)
        
    def _validate_state(self, batch_size):
        """Validate that internal state is properly initialized."""
        if self.syn_state is None:
            raise RuntimeError("syn_state is None. Call reset() before forward().")
        if self.syn_state.size(0) != batch_size:
            raise RuntimeError(
                f"Batch size mismatch: syn_state has {self.syn_state.size(0)} "
                f"but expected {batch_size}. Call reset() with correct batch size."
            )
    
    def forward(self, x):
        # Validate state before processing
        self._validate_state(x.size(0))
        z = self.enc(x)
        spk = self.lif(z)  # SpikingJelly neurons return spikes directly
        y = self.dec(spk)
        # More numerically stable EMA update
        self.syn_state = self.syn_alpha * self.syn_state + (1 - self.syn_alpha) * y
        return self.syn_state

class SelectiveSSMSpiking(nn.Module):
    """
    Mamba-style *selective* diagonal SSM core with:
    - selective Δt (per-dimension time step)
    - selective B (input-dependent scaling of injected input)
    - selective C (input-dependent, low-rank readout)

    The full derivations/diagrams and usage notes are maintained in `README.md`.
    """
    def __init__(self, in_dim: int, state_dim: int,
                 gate_hidden=128, dt_min=1e-3, dt_max=0.2,
                 readout_dim=64, feat_dim=16,
                 syn_alpha: float = 0.8,
                 gate_temp: float = 1.0,
                 lif_tau: float = 2.0,
                 lif_v_threshold: float = 1.0,
                 lif_v_reset: float = 0.0,
                 c_hidden: int = 64,
                 c_rank: int | None = None):
        super().__init__()
        self.N = state_dim
        self.dt_min, self.dt_max = dt_min, dt_max
        self.readout_dim = readout_dim
        self.gate_temp = gate_temp
        self.feat_dim = feat_dim  # Store feat_dim early

        taus = torch.logspace(start=torch.log10(torch.tensor(1.0)),
                              end=torch.log10(torch.tensor(784.0)),
                              steps=self.N)
        lambdas = -1.0 / taus
        lambdas = torch.clamp(lambdas, max=-1e-4)

        # Mamba-like diagonal A parameterization:
        #   A = -exp(logA)  (strictly negative, log-domain parameter)
        a_mag = torch.clamp(-lambdas, min=1e-6)
        self.logA = nn.Parameter(torch.log(a_mag))

        self.b = nn.Parameter(torch.ones(self.N))
        # Explicit input-dependent B(x) factor (Mamba-style "selective B").
        # This scales the injected input per state dimension.
        self.b_in = nn.Linear(feat_dim, self.N, bias=True)
        with torch.no_grad():
            nn.init.zeros_(self.b_in.weight)
            # Initialize so B(x) ≈ 1 at start (no scaling), for stable early training.
            # softplus_inv(1.0) gives bias such that softplus(bias) ~= 1.0
            nn.init.constant_(self.b_in.bias, float(softplus_inv(torch.tensor(1.0))))
        
        # REMOVED: self.c = nn.Parameter(...)
        # ADDED: C network for input-dependent readout with low-rank approximation
        # Configurable c_rank with sensible default
        if c_rank is None:
            self.c_rank = min(128, self.N // 2)
        else:
            self.c_rank = c_rank
        
        self.c_network = nn.Sequential(
            nn.Linear(self.feat_dim, c_hidden, bias=True),
            nn.LayerNorm(c_hidden),  # ADD THIS
            nn.SiLU(),
            nn.Linear(c_hidden, self.c_rank, bias=True),
            nn.Tanh()  # Keep values bounded
        )
        # Low-rank projection matrices
        self.c_proj_left = nn.Parameter(torch.randn(self.N, self.c_rank) * 0.1)
        self.c_proj_right = nn.Parameter(torch.randn(self.c_rank, self.readout_dim) * 0.1)
        
        # Initialize C network to produce reasonable initial values
        with torch.no_grad():
            # Target the last Linear layer (before Tanh)
            linear_layer = self.c_network[3]  # 4th element (index 3) is the Linear layer
            nn.init.normal_(linear_layer.weight, std=1.0 / math.sqrt(self.c_rank))
            nn.init.zeros_(linear_layer.bias)

        self.feat = nn.Sequential(
            nn.Linear(in_dim, feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.SiLU(),
        )
        self.x_proj = nn.Linear(feat_dim, feat_dim, bias=True)

        # Spiking-only: gates are always SpikingJelly LIF + EMA readout.
        self.gin = SpikingReadout(
            feat_dim, gate_hidden, 1,
            tau=lif_tau, v_threshold=lif_v_threshold, v_reset=lif_v_reset,
            syn_alpha=syn_alpha
        )
        # NOTE: gout removed since C is now fully selective
        self.dt = SpikingReadout(
            feat_dim, gate_hidden, self.N,
            tau=lif_tau, v_threshold=lif_v_threshold, v_reset=lif_v_reset,
            syn_alpha=syn_alpha
        )
        # Mamba-like dt parameterization: dt = softplus(dt_raw + dt_bias), then clamp.
        # Initialize dt_bias so that softplus(dt_bias) is near the midpoint of [dt_min, dt_max].
        dt_target = float((dt_min + dt_max) / 2.0)
        dt_bias_init = softplus_inv(torch.tensor(dt_target)).repeat(self.N)
        self.dt_bias = nn.Parameter(dt_bias_init)
                
        self.u_proj = nn.Linear(in_dim, self.N, bias=True)

        with torch.no_grad():
            # Start ~0.5 to avoid initial saturation
            if hasattr(self.gin, "dec") and hasattr(self.gin.dec, "bias"):
                nn.init.constant_(self.gin.dec.bias, 0.0)
            if hasattr(self.dt, "dec") and hasattr(self.dt.dec, "bias"):
                nn.init.constant_(self.dt.dec.bias, 0.0)

        # Recurrent state; do not serialize into checkpoints (batch-size dependent during training)
        self.register_buffer("s", torch.zeros(1, self.N), persistent=False)

    @staticmethod
    def _safe_beta(a, dt, eps=1e-6):
        """Stable β(dt) for diagonal SSM: β = (exp(a·dt) - 1)/a with β≈dt when a≈0."""
        x = a * dt
        num = torch.expm1(x)  # exp(x) - 1, more numerically stable
        den = torch.where(torch.abs(a) < eps, torch.ones_like(a), a)
        beta = num / den
        
        # Special case: when a ≈ 0, use β ≈ dt (from L'Hôpital's rule)
        return torch.where(torch.abs(a) < eps, dt, beta)

    def reset_state(self, B, device=None):
        self.s = torch.zeros(B, self.N, device=device or self.s.device)
        self.gin.reset(B, device)
        self.dt.reset(B, device)

    def step(self, x_t: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        """
        One recurrent update (single timestep) with selective Δt, B(x), and low-rank C(x).

        For the longer Algorithm-2 mapping notes, see `README.md`.
        """
        # Diagonal A parameterization: a = -exp(logA) (strictly negative for stability)
        a = -torch.exp(self.logA)  # [N]

        # Shared context features for all selective mechanisms (Δt, B(x), C(x))
        f = self.feat(x_t)  # [B, feat_dim]
        xmix = torch.tanh(self.x_proj(f))  # [B, feat_dim]

        # Selective input gate
        gin = torch.sigmoid(self.gin(xmix) / self.gate_temp)  # [B, 1] (broadcast later)

        # Selective time step (per state dimension)
        dt_raw = self.dt(xmix)  # [B, N]
        dt = F.softplus(dt_raw + self.dt_bias)  # [B, N]
        dt = dt.clamp(min=self.dt_min, max=self.dt_max)

        # Discrete-time SSM parameters
        phi = torch.exp(a * dt).clamp(max=1 - 1e-6)  # [B, N]
        beta = self._safe_beta(a, dt)  # [B, N]

        # Explicit B(x) selection (Mamba-style): inject via beta * (B(x) * u_proj(u))
        u_vec = self.u_proj(u_t)  # [B, N]
        b = self.b.unsqueeze(0).expand_as(beta)  # [B, N]
        b_sel = F.softplus(self.b_in(xmix))  # [B, N], initialized near 1.0
        Bx = b * b_sel * gin  # [B, N] (gin broadcasts from [B,1])
        inj = beta * (Bx * u_vec)  # [B, N]

        # Recurrent update
        self.s = phi * self.s + inj  # [B, N]

        # Selective readout via low-rank C(x)
        c_low_rank = self.c_network(xmix)  # [B, c_rank]
        y = torch.matmul(
            torch.matmul(self.s, self.c_proj_left) * c_low_rank, 
            self.c_proj_right
        )
        return y

    def forward_vectorized(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Vectorized forward pass for the entire sequence.
        Much faster than sequential step() calls.
        
        Args:
            seq: [T, B, 1] input sequence
            
        Returns:
            outputs: [T, B, readout_dim] sequence of outputs
        """
        T, B, _ = seq.shape
        device = seq.device
        
        # Pre-allocate output tensor
        outputs = torch.empty(T, B, self.readout_dim, device=device, dtype=seq.dtype)
        
        # Pre-compute all features and gates for the entire sequence
        # This allows for better vectorization and reduces redundant computation
        seq_flat = seq.reshape(-1, 1)  # [T*B, 1]
        
        # Run feature/gates/C precompute in full precision
        with torch.cuda.amp.autocast(enabled=False):
            f_all = self.feat(seq_flat.float())
            xmix_all = torch.tanh(self.x_proj(f_all))
            
            # Compute all gates and dt for the entire sequence
            # Reset readout networks for the full batch size
            self.gin.reset(T * B, device)
            self.dt.reset(T * B, device)
            
            gin_all = torch.sigmoid(self.gin(xmix_all).reshape(T, B, 1) / self.gate_temp)
            dt_raw_all = self.dt(xmix_all).reshape(T, B, self.N)
            dt_all = F.softplus(dt_raw_all + self.dt_bias.view(1, 1, self.N))
            dt_all = dt_all.clamp(min=self.dt_min, max=self.dt_max)
            b_sel_all = F.softplus(self.b_in(xmix_all)).reshape(T, B, self.N)
            c_low_rank_all = self.c_network(xmix_all).reshape(T, B, self.c_rank)
        
        # Pre-compute SSM parameters
        a = -torch.exp(self.logA)  # [N]
        phi_all = torch.exp(a.unsqueeze(0).unsqueeze(0) * dt_all).clamp(max=1 - 1e-6)
        beta_all = self._safe_beta(a.unsqueeze(0).unsqueeze(0), dt_all)  # [T, B, N]
        
        # Pre-compute input projections and explicit B(x)
        u_vec_all = self.u_proj(seq_flat).reshape(T, B, self.N)  # [T, B, N]
        b = self.b.unsqueeze(0).unsqueeze(0)  # [1, 1, N]
        Bx_all = b * b_sel_all * gin_all  # [T, B, N]
        inj_all = beta_all * (Bx_all * u_vec_all)  # [T, B, N]
        
        # Sequential state update (unavoidable due to recurrence)
        for t in range(T):
            # State update
            self.s = phi_all[t] * self.s + inj_all[t]  # [B, N]
            
            # C readout
            c_intermediate = torch.matmul(self.s, self.c_proj_left)  # [B, c_rank]
            c_weighted = c_intermediate * c_low_rank_all[t]  # [B, c_rank]
            outputs[t] = torch.matmul(c_weighted, self.c_proj_right)  # [B, readout_dim]
        
        return outputs

    @torch._dynamo.disable()
    def debug_peek(self, x_t: torch.Tensor) -> dict:
        # Eager-only, no graph capture, no grads
        with torch.no_grad():
            f = self.feat(x_t)
            xmix = torch.tanh(self.x_proj(f))
            
            # Temporarily reset readout networks for debug
            batch_size = x_t.size(0)
            self.gin.reset(batch_size, x_t.device)
            self.dt.reset(batch_size, x_t.device)
            
            gin = torch.sigmoid(self.gin(xmix) / self.gate_temp)

            dt_raw = self.dt(xmix)
            dt = F.softplus(dt_raw + self.dt_bias)
            dtN = dt.clamp(min=self.dt_min, max=self.dt_max)
            
            a = -torch.exp(self.logA)
            phi = torch.exp(a * dtN)

            b = self.b.unsqueeze(0).expand_as(phi)

            u_vec = self.u_proj(x_t)
            b_sel = F.softplus(self.b_in(xmix))
            Bx = b * b_sel * gin
            inj = self._safe_beta(a, dtN) * (Bx * u_vec)

            # C stats (low-rank)
            c_low_rank = self.c_network(xmix)
            c_norm = c_low_rank.norm(dim=-1).mean().item()

            return {
                "dt": dt.mean().item(),
                "gin": gin.mean().item(),
                "phi": phi.mean().item(),
                "inj": inj.abs().mean().item(),
                "c_norm": c_norm,
            }


class PMNISTModel(nn.Module):
    def __init__(self, state_dim=256, gate_hidden=128, dt_min=1e-3, dt_max=0.2,
                 readout_dim=64, last_k_avg=32, head_hidden=64, dropout=0.0,
                 feat_dim=16, syn_alpha=0.8, gate_temp=1.0,
                 lif_tau=2.0, lif_v_threshold=1.0, lif_v_reset=0.0, c_hidden=64,
                 c_rank=None):
        super().__init__()
        self.core = SelectiveSSMSpiking(in_dim=1, state_dim=state_dim,
                                        gate_hidden=gate_hidden,
                                        dt_min=dt_min, dt_max=dt_max,
                                        readout_dim=readout_dim,
                                        feat_dim=feat_dim,
                                        syn_alpha=syn_alpha,
                                        gate_temp=gate_temp,
                                        lif_tau=lif_tau,
                                        lif_v_threshold=lif_v_threshold,
                                        lif_v_reset=lif_v_reset,
                                        c_hidden=c_hidden,
                                        c_rank=c_rank)
        self.last_k_avg = max(1, int(last_k_avg))
        self.norm = nn.LayerNorm(readout_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(readout_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 10),
        )
    def reset_state(self, B, device=None):
        self.core.reset_state(B, device)
    
    def forward(self, seq):  # [T,B,1]
        """
        Main forward pass.
        Note: reset_state must be called before forward() by the caller.
        """
        return self.forward_vectorized(seq)

    def forward_vectorized(self, seq: torch.Tensor) -> torch.Tensor:
        """Vectorized logits path; expects seq [T,B,1]."""
        T, _, _ = seq.shape
        feats = self.core.forward_vectorized(seq)
        if self.last_k_avg > 1:
            k = min(self.last_k_avg, T)
            feats = feats[-k:].mean(dim=0)
        else:
            feats = feats[-1]
        feats = self.norm(feats)
        return self.head(feats)

    def forward_stepwise(self, seq: torch.Tensor) -> torch.Tensor:
        """Stepwise logits path using core.step(); expects seq [T,B,1]."""
        T, B, _ = seq.shape
        # Accumulate readout features over time
        outputs = []
        for t in range(T):
            y_t = self.core.step(seq[t], seq[t])  # x_t and u_t are the same scalar stream
            outputs.append(y_t)
        feats = torch.stack(outputs, dim=0)  # [T,B,readout_dim]
        if self.last_k_avg > 1:
            k = min(self.last_k_avg, T)
            feats = feats[-k:].mean(dim=0)
        else:
            feats = feats[-1]
        feats = self.norm(feats)
        return self.head(feats)

# -----------------------------
# Phase 1: Stacked SSM Architecture (for 90%+ accuracy)
# -----------------------------

class SelectiveSSMSpikingWithConv(nn.Module):
    """
    Enhanced SelectiveSSMSpiking with depthwise temporal convolution.
    
    The conv provides local temporal context before selection, which helps
    the gates make better decisions (they see a window, not just current pixel).
    
    Optionally includes z-gate (output gating) for Mamba-style output control.
    """
    def __init__(self, in_dim: int, state_dim: int,
                 gate_hidden=128, dt_min=1e-3, dt_max=0.2,
                 readout_dim=64, feat_dim=16,
                 syn_alpha: float = 0.8,
                 gate_temp: float = 1.0,
                 lif_tau: float = 2.0,
                 lif_v_threshold: float = 1.0,
                 lif_v_reset: float = 0.0,
                 c_hidden: int = 64,
                 c_rank: int | None = None,
                 conv_kernel: int = 4,
                 use_z_gate: bool = False):
        super().__init__()
        self.in_dim = in_dim
        self.N = state_dim
        self.dt_min, self.dt_max = dt_min, dt_max
        self.readout_dim = readout_dim
        self.gate_temp = gate_temp
        self.feat_dim = feat_dim
        self.conv_kernel = conv_kernel
        self.use_z_gate = use_z_gate

        # Depthwise temporal convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=in_dim,
            out_channels=in_dim,
            kernel_size=conv_kernel,
            groups=in_dim,  # depthwise
            padding=conv_kernel - 1,  # causal padding
            bias=True
        )

        # Initialize logA from log-spaced time constants
        taus = torch.logspace(start=torch.log10(torch.tensor(1.0)),
                              end=torch.log10(torch.tensor(784.0)),
                              steps=self.N)
        lambdas = -1.0 / taus
        lambdas = torch.clamp(lambdas, max=-1e-4)
        a_mag = torch.clamp(-lambdas, min=1e-6)
        self.logA = nn.Parameter(torch.log(a_mag))

        self.b = nn.Parameter(torch.ones(self.N))
        self.b_in = nn.Linear(feat_dim, self.N, bias=True)
        with torch.no_grad():
            nn.init.zeros_(self.b_in.weight)
            nn.init.constant_(self.b_in.bias, float(softplus_inv(torch.tensor(1.0))))
        
        if c_rank is None:
            self.c_rank = min(128, self.N // 2)
        else:
            self.c_rank = c_rank
        
        self.c_network = nn.Sequential(
            nn.Linear(self.feat_dim, c_hidden, bias=True),
            nn.LayerNorm(c_hidden),
            nn.SiLU(),
            nn.Linear(c_hidden, self.c_rank, bias=True),
            nn.Tanh()
        )
        self.c_proj_left = nn.Parameter(torch.randn(self.N, self.c_rank) * 0.1)
        self.c_proj_right = nn.Parameter(torch.randn(self.c_rank, self.readout_dim) * 0.1)
        
        with torch.no_grad():
            linear_layer = self.c_network[3]
            nn.init.normal_(linear_layer.weight, std=1.0 / math.sqrt(self.c_rank))
            nn.init.zeros_(linear_layer.bias)

        self.feat = nn.Sequential(
            nn.Linear(in_dim, feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.SiLU(),
        )
        self.x_proj = nn.Linear(feat_dim, feat_dim, bias=True)

        self.gin = SpikingReadout(
            feat_dim, gate_hidden, 1,
            tau=lif_tau, v_threshold=lif_v_threshold, v_reset=lif_v_reset,
            syn_alpha=syn_alpha
        )
        self.dt = SpikingReadout(
            feat_dim, gate_hidden, self.N,
            tau=lif_tau, v_threshold=lif_v_threshold, v_reset=lif_v_reset,
            syn_alpha=syn_alpha
        )
        dt_target = float((dt_min + dt_max) / 2.0)
        dt_bias_init = softplus_inv(torch.tensor(dt_target)).repeat(self.N)
        self.dt_bias = nn.Parameter(dt_bias_init)
                
        self.u_proj = nn.Linear(in_dim, self.N, bias=True)

        with torch.no_grad():
            if hasattr(self.gin, "dec") and hasattr(self.gin.dec, "bias"):
                nn.init.constant_(self.gin.dec.bias, 0.0)
            if hasattr(self.dt, "dec") and hasattr(self.dt.dec, "bias"):
                nn.init.constant_(self.dt.dec.bias, 0.0)

        # Z-gate for output gating (Mamba-style)
        if self.use_z_gate:
            self.z_gate = SpikingReadout(
                feat_dim, gate_hidden, readout_dim,
                tau=lif_tau, v_threshold=lif_v_threshold, v_reset=lif_v_reset,
                syn_alpha=syn_alpha
            )
            with torch.no_grad():
                if hasattr(self.z_gate, "dec") and hasattr(self.z_gate.dec, "bias"):
                    nn.init.constant_(self.z_gate.dec.bias, 0.0)

        # Recurrent state; do not serialize into checkpoints (batch-size dependent during training)
        self.register_buffer("s", torch.zeros(1, self.N), persistent=False)

    @staticmethod
    def _safe_beta(a, dt, eps=1e-6):
        x = a * dt
        num = torch.expm1(x)
        den = torch.where(torch.abs(a) < eps, torch.ones_like(a), a)
        beta = num / den
        return torch.where(torch.abs(a) < eps, dt, beta)

    def reset_state(self, B, device=None):
        self.s = torch.zeros(B, self.N, device=device or self.s.device)
        self.gin.reset(B, device)
        self.dt.reset(B, device)
        if self.use_z_gate:
            self.z_gate.reset(B, device)

    def forward_vectorized(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Vectorized forward with depthwise conv for local context.
        
        Args:
            seq: [T, B, in_dim] input sequence
            
        Returns:
            outputs: [T, B, readout_dim] sequence of outputs
        """
        T, B, D = seq.shape
        device = seq.device
        
        # Apply causal depthwise conv for local temporal context
        seq_for_conv = seq.permute(1, 2, 0)  # [B, in_dim, T]
        seq_conv = self.conv1d(seq_for_conv)[:, :, :T]  # causal: keep only first T
        seq_conv = seq_conv.permute(2, 0, 1)  # [T, B, in_dim]
        
        outputs = torch.empty(T, B, self.readout_dim, device=device, dtype=seq.dtype)
        
        seq_flat = seq_conv.reshape(-1, D)  # [T*B, in_dim]
        
        with torch.cuda.amp.autocast(enabled=False):
            f_all = self.feat(seq_flat.float())
            xmix_all = torch.tanh(self.x_proj(f_all))
            
            self.gin.reset(T * B, device)
            self.dt.reset(T * B, device)
            
            gin_all = torch.sigmoid(self.gin(xmix_all).reshape(T, B, 1) / self.gate_temp)
            dt_raw_all = self.dt(xmix_all).reshape(T, B, self.N)
            dt_all = F.softplus(dt_raw_all + self.dt_bias.view(1, 1, self.N))
            dt_all = dt_all.clamp(min=self.dt_min, max=self.dt_max)
            b_sel_all = F.softplus(self.b_in(xmix_all)).reshape(T, B, self.N)
            c_low_rank_all = self.c_network(xmix_all).reshape(T, B, self.c_rank)
            
            # Z-gate for output gating (Mamba-style)
            if self.use_z_gate:
                self.z_gate.reset(T * B, device)
                z_all = torch.sigmoid(self.z_gate(xmix_all).reshape(T, B, self.readout_dim) / self.gate_temp)
        
        a = -torch.exp(self.logA)
        phi_all = torch.exp(a.unsqueeze(0).unsqueeze(0) * dt_all).clamp(max=1 - 1e-6)
        beta_all = self._safe_beta(a.unsqueeze(0).unsqueeze(0), dt_all)
        
        # Use original seq (not conv) for input projection
        u_vec_all = self.u_proj(seq.reshape(-1, D)).reshape(T, B, self.N)
        b = self.b.unsqueeze(0).unsqueeze(0)
        Bx_all = b * b_sel_all * gin_all
        inj_all = beta_all * (Bx_all * u_vec_all)
        
        for t in range(T):
            self.s = phi_all[t] * self.s + inj_all[t]
            c_intermediate = torch.matmul(self.s, self.c_proj_left)
            c_weighted = c_intermediate * c_low_rank_all[t]
            y_t = torch.matmul(c_weighted, self.c_proj_right)
            # Apply z-gate if enabled
            if self.use_z_gate:
                y_t = y_t * z_all[t]
            outputs[t] = y_t
        
        return outputs


class SpikingSSMBlock(nn.Module):
    """
    One SSM block with pre-norm and residual connection.
    
    This is the building block for the stacked architecture.
    Uses LayerNorm before SSM (pre-norm) and adds residual after.
    """
    def __init__(self, d_model: int, state_dim: int,
                 gate_hidden: int = 128, dt_min: float = 1e-3, dt_max: float = 0.2,
                 feat_dim: int = 16, syn_alpha: float = 0.8, gate_temp: float = 1.0,
                 lif_tau: float = 2.0, lif_v_threshold: float = 1.0,
                 lif_v_reset: float = 0.0, c_hidden: int = 64, c_rank: int | None = None,
                 conv_kernel: int = 4, dropout: float = 0.1, use_z_gate: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSMSpikingWithConv(
            in_dim=d_model, state_dim=state_dim,
            gate_hidden=gate_hidden, dt_min=dt_min, dt_max=dt_max,
            readout_dim=d_model,  # output same dim for residual
            feat_dim=feat_dim, syn_alpha=syn_alpha, gate_temp=gate_temp,
            lif_tau=lif_tau, lif_v_threshold=lif_v_threshold,
            lif_v_reset=lif_v_reset, c_hidden=c_hidden, c_rank=c_rank,
            conv_kernel=conv_kernel, use_z_gate=use_z_gate
        )
        self.dropout = nn.Dropout(dropout)
    
    def reset_state(self, B, device=None):
        self.ssm.reset_state(B, device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [T, B, d_model]
        Returns:
            [T, B, d_model]
        """
        residual = x
        # Pre-norm: normalize over d_model dimension
        x_normed = self.norm(x)  # LayerNorm works on last dim
        x_ssm = self.ssm.forward_vectorized(x_normed)
        return residual + self.dropout(x_ssm)


class StackedSpikingSSM(nn.Module):
    """
    Multi-layer stacked SSM with input projection.
    
    This is the core of Phase 1: stack multiple SSM blocks with residuals
    and add an input projection to map scalar pixels to d_model dimensions.
    """
    def __init__(self, n_layers: int = 4, d_model: int = 128, state_dim: int = 256,
                 gate_hidden: int = 128, dt_min: float = 1e-3, dt_max: float = 0.2,
                 feat_dim: int = 16, syn_alpha: float = 0.8, gate_temp: float = 1.0,
                 lif_tau: float = 2.0, lif_v_threshold: float = 1.0,
                 lif_v_reset: float = 0.0, c_hidden: int = 64, c_rank: int | None = None,
                 conv_kernel: int = 4, dropout: float = 0.1, use_z_gate: bool = False):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Input projection: pixel scalar → d_model
        self.input_proj = nn.Linear(1, d_model)
        
        # Stack of SSM blocks
        self.layers = nn.ModuleList([
            SpikingSSMBlock(
                d_model=d_model, state_dim=state_dim,
                gate_hidden=gate_hidden, dt_min=dt_min, dt_max=dt_max,
                feat_dim=feat_dim, syn_alpha=syn_alpha, gate_temp=gate_temp,
                lif_tau=lif_tau, lif_v_threshold=lif_v_threshold,
                lif_v_reset=lif_v_reset, c_hidden=c_hidden, c_rank=c_rank,
                conv_kernel=conv_kernel, dropout=dropout, use_z_gate=use_z_gate
            ) for _ in range(n_layers)
        ])
        
        # Final normalization
        self.final_norm = nn.LayerNorm(d_model)
    
    def reset_state(self, B, device=None):
        for layer in self.layers:
            layer.reset_state(B, device)
    
    def forward_vectorized(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq: [T, B, 1] input sequence (raw pixels)
        Returns:
            [T, B, d_model] output sequence
        """
        # Project input to d_model
        x = self.input_proj(seq)  # [T, B, d_model]
        
        # Pass through all layers
        for layer in self.layers:
            x = layer(x)
        
        # Final normalization
        return self.final_norm(x)


class StackedPMNISTModel(nn.Module):
    """
    pMNIST model using stacked SSM architecture (Phase 1).
    
    This replaces the single-layer PMNISTModel for higher accuracy.
    """
    def __init__(self, n_layers: int = 4, d_model: int = 128, state_dim: int = 256,
                 gate_hidden: int = 128, dt_min: float = 1e-3, dt_max: float = 0.2,
                 feat_dim: int = 16, syn_alpha: float = 0.8, gate_temp: float = 1.0,
                 lif_tau: float = 2.0, lif_v_threshold: float = 1.0,
                 lif_v_reset: float = 0.0, c_hidden: int = 64, c_rank: int | None = None,
                 conv_kernel: int = 4, dropout: float = 0.1,
                 last_k_avg: int = 32, head_hidden: int = 64, use_z_gate: bool = False):
        super().__init__()
        self.core = StackedSpikingSSM(
            n_layers=n_layers, d_model=d_model, state_dim=state_dim,
            gate_hidden=gate_hidden, dt_min=dt_min, dt_max=dt_max,
            feat_dim=feat_dim, syn_alpha=syn_alpha, gate_temp=gate_temp,
            lif_tau=lif_tau, lif_v_threshold=lif_v_threshold,
            lif_v_reset=lif_v_reset, c_hidden=c_hidden, c_rank=c_rank,
            conv_kernel=conv_kernel, dropout=dropout, use_z_gate=use_z_gate
        )
        self.last_k_avg = max(1, int(last_k_avg))
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 10),
        )
    
    def reset_state(self, B, device=None):
        self.core.reset_state(B, device)
    
    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq: [T, B, 1] input sequence
        Returns:
            [B, 10] logits
        """
        T, B, _ = seq.shape
        
        feats = self.core.forward_vectorized(seq)  # [T, B, d_model]
        
        if self.last_k_avg > 1:
            k = min(self.last_k_avg, T)
            feats = feats[-k:].mean(dim=0)  # [B, d_model]
        else:
            feats = feats[-1]  # [B, d_model]
        
        return self.head(feats)


# -----------------------------
# Training
# -----------------------------
@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    clip: float = 1.0
    warmup_epochs: int = 3
    # LR schedule:
    # - cosine: linear warmup (epochs) then CosineAnnealingLR down to min_lr
    # - onecycle: OneCycleLR stepped per-batch (warmup_epochs ignored)
    # - none: constant lr (optionally with warmup_epochs)
    scheduler: str = "cosine"  # cosine|onecycle|none
    min_lr: float = 0.0  # cosine eta_min
    onecycle_pct_start: float = 0.1
    onecycle_div_factor: float = 25.0
    onecycle_final_div_factor: float = 1e4
    patience: int = 5
    state_dim: int = 256
    gate_hidden: int = 128
    dt_min: float = 1e-3
    dt_max: float = 0.2
    readout_dim: int = 64
    last_k_avg: int = 32
    head_hidden: int = 64
    dropout: float = 0.0
    feat_dim: int = 16
    perm_seed: int = 0
    seed: int = DEFAULT_SEED
    quick: bool = False
    enable_cudagraphs: bool = False
    amp_bf16: bool = False
    cache_dir: str = "./data/PMNIST"
    rebuild_cache: bool = False
    resume: str | None = None
    # Resume behavior controls (useful for fine-tuning)
    resume_reset_optimizer: bool = False  # if True, ignore saved optimizer state on resume
    resume_reset_scheduler: bool = False  # if True, ignore saved scheduler state on resume
    resume_override_lr: bool = False  # if True, force cfg.lr after loading optimizer state
    resume_override_weight_decay: bool = False  # if True, force cfg.weight_decay after loading optimizer state
    checkpoint_dir: str = "checkpoints"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    syn_alpha: float = 0.8
    gate_temp: float = 1.0
    input_norm: bool = True
    input_mean: float = 0.0
    input_std: float = 1.0
    lif_tau: float = 2.0
    lif_v_threshold: float = 1.0
    lif_v_reset: float = 0.0
    c_hidden: int = 64  # hidden size for C network
    c_rank: int | None = None  # rank for C low-rank decomposition
    # Phase 1: Stacked model configuration
    use_stacked: bool = False  # Use stacked SSM architecture
    n_layers: int = 4  # Number of SSM layers (when use_stacked=True)
    d_model: int = 128  # Model dimension (when use_stacked=True)
    conv_kernel: int = 4  # Depthwise conv kernel size (when use_stacked=True)
    # Phase 2: Z-gate (output gating)
    use_z_gate: bool = False  # Enable Mamba-style output gating
    # Phase 3: Complex S4D-C states
    use_complex: bool = False  # Enable complex-valued S4D-C diagonal states
    # Regularization
    label_smoothing: float = 0.0  # Label smoothing for cross-entropy loss
    pixel_noise: float = 0.0  # Gaussian noise std to add to pixels during training

# -----------------------------
# Runtime config, utils & data cache (kept below class definitions)
# -----------------------------
def set_cudagraphs(enabled: bool):
    try:
        import torch._inductor.config as inductor_config  # type: ignore
        inductor_config.triton.cudagraphs = enabled
    except Exception:
        pass
    if not enabled:
        os.environ["TORCHINDUCTOR_DISABLE_CUDAGRAPHS"] = "1"
    else:
        os.environ.pop("TORCHINDUCTOR_DISABLE_CUDAGRAPHS", None)

# Default: off (train_pmnist can enable it based on cfg.enable_cudagraphs)
set_cudagraphs(False)

# Speed knobs
import torch.backends.cudnn as cudnn
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
cudnn.benchmark = True
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.accumulated_cache_size_limit = 512
    torch._dynamo.config.assume_static_by_default = True
except Exception:
    pass

if torch.cuda.is_available():
    maj, minr = torch.cuda.get_device_capability()
    te = (torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16)
    print(f"CUDA capability: sm_{maj}{minr} | AMP dtype: {te}")

def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_permute(seed=0) -> torch.Tensor:
    """
    Return the pixel-order permutation for (p)MNIST.

    Convention:
    - seed >= 0 : permuted MNIST (pMNIST) with a deterministic random permutation
    - seed < 0  : sequential MNIST (sMNIST) / identity permutation
    """
    n = 28 * 28
    if int(seed) < 0:
        return torch.arange(n)
    g = torch.Generator().manual_seed(int(seed))
    return torch.randperm(n, generator=g)

def _download_mnist():
    (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.mnist.load_data()
    x_tr = (x_tr.astype("float32") / 255.0)
    x_te = (x_te.astype("float32") / 255.0)
    return {"images": x_tr, "labels": y_tr}, {"images": x_te, "labels": y_te}

def _mnist_to_pmnist_tensors(ds, perm: torch.Tensor):
    imgs = ds["images"]
    labels = ds["labels"]
    N = imgs.shape[0]
    perm_np = perm.cpu().numpy()
    x = imgs.reshape(N, -1)[:, perm_np]
    x = torch.from_numpy(x).unsqueeze(-1).contiguous()
    y = torch.from_numpy(labels.astype("int64")).contiguous()
    return x, y

def build_or_load_pmnist(cache_dir: str, perm_seed: int, rebuild: bool = False) -> tuple[str, str]:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    is_sequential = int(perm_seed) < 0
    if is_sequential:
        # Use distinct filenames so we never accidentally reuse older pmnist_perm-1 caches.
        train_path = cache / "smnist_train.pt"
        test_path = cache / "smnist_test.pt"
        tag = "SMNIST"
    else:
        train_path = cache / f"pmnist_perm{perm_seed}_train.pt"
        test_path = cache / f"pmnist_perm{perm_seed}_test.pt"
        tag = "PMNIST"
    if not rebuild and train_path.exists() and test_path.exists():
        print(f"[{tag}] Using cached tensors in: {cache.resolve()}")
        return str(train_path), str(test_path)
    if is_sequential:
        print(f"[{tag}] Building cache at: {cache.resolve()} (sequential / identity permutation)")
    else:
        print(f"[{tag}] Building cache at: {cache.resolve()} (perm_seed={perm_seed})")
    perm = make_permute(perm_seed)
    train_raw, test_raw = _download_mnist()
    x_tr, y_tr = _mnist_to_pmnist_tensors(train_raw, perm)
    x_te, y_te = _mnist_to_pmnist_tensors(test_raw, perm)
    torch.save(
        {"x": x_tr, "y": y_tr, "perm_seed": perm_seed, "perm": perm, "is_sequential": is_sequential},
        train_path,
    )
    torch.save(
        {"x": x_te, "y": y_te, "perm_seed": perm_seed, "perm": perm, "is_sequential": is_sequential},
        test_path,
    )
    print(f"[{tag}] Saved: {train_path.name} ({x_tr.shape[0]}), {test_path.name} ({x_te.shape[0]})")
    return str(train_path), str(test_path)

def collate_pmnist(batch) -> tuple[torch.Tensor, torch.Tensor]:
    """Default pMNIST collation: keep full 784-step sequence, no downsampling."""
    xs, ys = zip(*batch)
    x = torch.stack(xs, dim=0)  # [B, 784, 1]
    y = torch.tensor(ys, dtype=torch.long)
    return x, y

def plot_training_curves(history: dict, save_path: Path, title: str = "Training Convergence"):
    """
    Plot training and test accuracy/loss curves.
    
    Args:
        history: Dictionary with 'epochs', 'train_loss', 'train_acc', 'test_loss', 'test_acc'
        save_path: Path to save the plot
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = history['epochs']
    
    # Plot 1: Accuracy
    ax1.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2, alpha=0.8)
    ax1.plot(epochs, history['test_acc'], 'r-', label='Test Accuracy', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title(f'{title} - Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Add best test accuracy marker
    best_idx = history['test_acc'].index(max(history['test_acc']))
    best_epoch = epochs[best_idx]
    best_acc = history['test_acc'][best_idx]
    ax1.plot(best_epoch, best_acc, 'r*', markersize=15, 
             label=f'Best: {best_acc:.2f}% (ep {best_epoch})')
    ax1.legend(fontsize=11)
    
    # Plot 2: Loss
    ax2.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2, alpha=0.8)
    ax2.plot(epochs, history['test_loss'], 'r-', label='Test Loss', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title(f'{title} - Loss', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved convergence plot: {save_path}")


def train_pmnist(cfg: TrainConfig) -> float:
    set_seed(cfg.seed)
    set_cudagraphs(cfg.enable_cudagraphs)
    device = torch.device(cfg.device)
    print(f"Device: {device} (cuda available={torch.cuda.is_available()})")
    if cfg.amp_bf16 and device.type != "cuda":
        print("[AMP] --amp-bf16 requested but device is not CUDA; disabling AMP.")
        cfg.amp_bf16 = False
    # Compute actual c_rank that will be used
    actual_c_rank = cfg.c_rank if cfg.c_rank is not None else min(128, cfg.state_dim // 2)
    print(f"[SELECTIVE C] Using input-dependent C matrix (c_hidden={cfg.c_hidden}, c_rank={actual_c_rank})")
    
    # Training configuration summary
    print(f"\n{'='*60}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*60}")
    if cfg.use_stacked:
        z_gate_str = " | z-gate: ON" if cfg.use_z_gate else ""
        print(f"[STACKED MODE] n_layers: {cfg.n_layers} | d_model: {cfg.d_model} | conv_kernel: {cfg.conv_kernel}{z_gate_str}")
    else:
        print("[SINGLE-LAYER MODE]")
    print(f"Epochs: {cfg.epochs} | Batch size: {cfg.batch_size} | Learning rate: {cfg.lr}")
    print(f"State dim: {cfg.state_dim} | Readout dim: {cfg.readout_dim} | Last k avg: {cfg.last_k_avg}")
    print(f"Gate hidden: {cfg.gate_hidden} | Feat dim: {cfg.feat_dim} | C hidden: {cfg.c_hidden} | C rank: {actual_c_rank}")
    print(f"Gates: spiking (SpikingJelly LIF) | Syn alpha: {cfg.syn_alpha} | Gate temp: {cfg.gate_temp}")
    print(f"DT range: [{cfg.dt_min:.4f}, {cfg.dt_max:.4f}] | Dropout: {cfg.dropout}")
    print(f"Label smoothing: {cfg.label_smoothing} | Pixel noise: {cfg.pixel_noise}")
    print(f"AMP bf16: {cfg.amp_bf16}")
    print(f"Input norm: {cfg.input_norm} | Seed: {cfg.seed} | Quick mode: {cfg.quick}")
    print(f"{'='*60}\n")

    train_file, test_file = build_or_load_pmnist(cfg.cache_dir, cfg.perm_seed, cfg.rebuild_cache)
    train_ds = PMNISTCached(train_file, expected_perm_seed=cfg.perm_seed)
    test_ds = PMNISTCached(test_file, expected_perm_seed=cfg.perm_seed)
    print(f"[PMNIST] Loaded cache: perm_seed={train_ds.perm_seed}, train={len(train_ds)}, test={len(test_ds)}")

    if cfg.quick:
        train_ds = Subset(train_ds, range(300))
        test_ds = Subset(test_ds, range(100))
        print("Running QUICK mode on subsets.")

    # --- Compute dataset mean/std for normalization (once) ---
    if cfg.input_norm:
        with torch.no_grad():
            # Handle both original dataset and Subset wrapper
            base_ds = train_ds.dataset if hasattr(train_ds, 'dataset') else train_ds
            flat = base_ds.x.view(-1).float()
            cfg.input_mean = flat.mean().item()
            cfg.input_std = flat.std().clamp_min(1e-6).item()
        print(f"[PMNIST] Input normalization: mean={cfg.input_mean:.4f}, std={cfg.input_std:.4f}")

    # Optimized data loading settings
    num_workers = int(os.environ.get("NUM_WORKERS", min(4, (os.cpu_count() or 2))))  # Reduced for stability
    pin = torch.cuda.is_available()
    use_persist = num_workers > 0 and not cfg.quick  # Disable for quick mode

    drop_last = False if cfg.quick else True

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=use_persist, prefetch_factor=4 if use_persist else None,  # Increased prefetch
        drop_last=drop_last, collate_fn=collate_pmnist
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=use_persist, prefetch_factor=4 if use_persist else None,  # Increased prefetch
        drop_last=drop_last, collate_fn=collate_pmnist
    )

    # Build model based on configuration
    if cfg.use_stacked:
        model = StackedPMNISTModel(
            n_layers=cfg.n_layers, d_model=cfg.d_model, state_dim=cfg.state_dim,
            gate_hidden=cfg.gate_hidden, dt_min=cfg.dt_min, dt_max=cfg.dt_max,
            feat_dim=cfg.feat_dim, syn_alpha=cfg.syn_alpha, gate_temp=cfg.gate_temp,
            lif_tau=cfg.lif_tau, lif_v_threshold=cfg.lif_v_threshold,
            lif_v_reset=cfg.lif_v_reset, c_hidden=cfg.c_hidden, c_rank=cfg.c_rank,
            conv_kernel=cfg.conv_kernel, dropout=cfg.dropout,
            last_k_avg=cfg.last_k_avg, head_hidden=cfg.head_hidden,
            use_z_gate=cfg.use_z_gate,
            use_complex=cfg.use_complex,
        ).to(device)
        z_gate_str = " + z-gate" if cfg.use_z_gate else ""
        print(f"[Model] Using StackedPMNISTModel with {cfg.n_layers} layers{z_gate_str}")
    else:
        model = PMNISTModel(state_dim=cfg.state_dim, gate_hidden=cfg.gate_hidden,
                            dt_min=cfg.dt_min, dt_max=cfg.dt_max,
                            readout_dim=cfg.readout_dim, last_k_avg=cfg.last_k_avg,
                            head_hidden=cfg.head_hidden, dropout=cfg.dropout,
                            feat_dim=cfg.feat_dim,
                            syn_alpha=cfg.syn_alpha, gate_temp=cfg.gate_temp,
                            lif_tau=cfg.lif_tau, lif_v_threshold=cfg.lif_v_threshold,
                            lif_v_reset=cfg.lif_v_reset, c_hidden=cfg.c_hidden,
                            c_rank=cfg.c_rank).to(device)
        print("[Model] Using single-layer PMNISTModel")

    # Parity check (only for single-layer model, stacked doesn't have forward_stepwise)
    if not cfg.use_stacked:
        try:
            model.eval()
            with torch.no_grad():
                x = torch.randn(12, 3, 1, device=device)
                model.reset_state(x.size(1), device)
                s1 = model.forward_stepwise(x)
                model.reset_state(x.size(1), device)
                s2 = model.forward_vectorized(x)
                diff = (s1 - s2).abs().max().item()
                print("[Parity] stepwise vs vectorized maxdiff:", diff)
        except Exception as e:
            print("[Parity] Skipped due to:", str(e))

    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable params: {trainable_params:,} | Total params: {total_params:,}")
    
    # Print C network stats (different structure for stacked vs single)
    if cfg.use_stacked:
        # Sum C network params across all layers
        c_total = 0
        for layer in model.core.layers:
            c_total += sum(p.numel() for p in layer.ssm.c_network.parameters())
            c_total += layer.ssm.c_proj_left.numel() + layer.ssm.c_proj_right.numel()
        print(f"[Model] C params (all layers): {c_total:,} ({100*c_total/total_params:.1f}%)")
    else:
        c_network_params = sum(p.numel() for p in model.core.c_network.parameters())
        c_proj_params = model.core.c_proj_left.numel() + model.core.c_proj_right.numel()
        c_total_params = c_network_params + c_proj_params
        print(f"[Model] C network params: {c_network_params:,} | C proj params: {c_proj_params:,} | C total: {c_total_params:,} ({100*c_total_params/total_params:.1f}%)")
    
    # Memory usage estimation
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        print(f"[Memory] GPU allocated: {memory_allocated:.2f} GB | Reserved: {memory_reserved:.2f} GB")
    
    # Data loading info
    print(f"[Data] Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")
    print(f"[Data] Workers: {num_workers} | Pin memory: {pin} | Persistent: {use_persist}")
    print(f"[Data] Prefetch factor: {4 if use_persist else 'None'}")

    # Compilation disabled by default for optimal performance
    # torch.compile has compatibility issues with SpikingJelly's reset_net()
    # Vectorized forward pass provides better performance without compilation
    print("Using vectorized forward pass (no compilation) for optimal performance")
    
    print("\nStarting training...")
    print(f"{'='*60}")

    # Exclude biases/LayerNorm from weight decay
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.ndim == 1 or n.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr
    )

    sched_mode = (getattr(cfg, "scheduler", "cosine") or "cosine").lower()
    base_sched: torch.optim.lr_scheduler._LRScheduler | None = None
    sched_step_per_batch = False
    if sched_mode == "cosine":
        t_max = max(1, cfg.epochs - cfg.warmup_epochs)
        base_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=t_max,
            eta_min=float(getattr(cfg, "min_lr", 0.0)),
        )
    elif sched_mode == "onecycle":
        # OneCycleLR is stepped per *optimizer step* (per batch).
        steps_per_epoch = len(train_loader)
        total_steps = max(1, cfg.epochs * steps_per_epoch)
        base_sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=cfg.lr,
            total_steps=total_steps,
            pct_start=float(getattr(cfg, "onecycle_pct_start", 0.1)),
            div_factor=float(getattr(cfg, "onecycle_div_factor", 25.0)),
            final_div_factor=float(getattr(cfg, "onecycle_final_div_factor", 1e4)),
            anneal_strategy="cos",
        )
        sched_step_per_batch = True
        if cfg.warmup_epochs:
            print("[Sched] onecycle ignores --warmup-epochs (already includes warmup). Consider --warmup-epochs 0.")
    elif sched_mode == "none":
        base_sched = None
    else:
        raise ValueError(f"Unknown scheduler '{cfg.scheduler}'. Use one of: cosine, onecycle, none.")

    def step_sched_epoch(epoch: int):
        # Epoch-level scheduler step (cosine/none). OneCycle is stepped per-batch.
        if sched_mode == "cosine":
            if epoch < cfg.warmup_epochs:
                warmup_factor = (epoch + 1) / max(1, cfg.warmup_epochs)
                for g in opt.param_groups:
                    g["lr"] = cfg.lr * warmup_factor
            else:
                assert base_sched is not None
                base_sched.step()
        elif sched_mode == "none":
            if epoch < cfg.warmup_epochs:
                warmup_factor = (epoch + 1) / max(1, cfg.warmup_epochs)
                for g in opt.param_groups:
                    g["lr"] = cfg.lr * warmup_factor
        else:
            # onecycle: no epoch-level stepping
            return

    def step_sched_batch():
        if sched_step_per_batch and base_sched is not None:
            base_sched.step()

    # Resume from checkpoint if specified
    start_epoch = 1
    best = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {
        'epochs': [],
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
    }
    
    if cfg.resume:
        print(f"\n{'='*60}")
        print(f"RESUMING FROM CHECKPOINT: {cfg.resume}")
        print(f"{'='*60}")
        
        checkpoint = torch.load(cfg.resume, map_location=device)
        
        # Load model state
        # NOTE: filter out recurrent state buffers (e.g., keys ending with ".s") which may have
        # batch-size-dependent shapes in older checkpoints.
        model_state = checkpoint.get('model_state_dict', {})
        if isinstance(model_state, dict):
            filtered_state = {k: v for k, v in model_state.items() if not k.endswith('.s')}
            dropped = len(model_state) - len(filtered_state)
        else:
            filtered_state = model_state
            dropped = 0

        incompatible = model.load_state_dict(filtered_state, strict=False)
        if dropped:
            print(f"✓ Loaded model state from epoch {checkpoint['epoch']} (dropped {dropped} state buffer keys)")
        else:
            print(f"✓ Loaded model state from epoch {checkpoint['epoch']}")
        if getattr(incompatible, "missing_keys", None) or getattr(incompatible, "unexpected_keys", None):
            mk = getattr(incompatible, "missing_keys", [])
            uk = getattr(incompatible, "unexpected_keys", [])
            if mk:
                print(f"  [Resume] Missing keys (ignored): {len(mk)}")
            if uk:
                print(f"  [Resume] Unexpected keys (ignored): {len(uk)}")
        
        # Restore training state
        start_epoch = checkpoint['epoch'] + 1
        # Backward compatibility: older checkpoints used 'best_acc' not 'best_test_acc'
        best = checkpoint.get('best_test_acc', checkpoint.get('best_acc', 0.0))
        best_epoch = checkpoint.get('best_epoch', checkpoint.get('best_ep', 0))
        patience_counter = 0  # Reset patience counter on resume

        # Load / reset optimizer state
        if cfg.resume_reset_optimizer:
            print("↻ Resume: resetting optimizer state (fine-tune mode)")
        else:
            opt.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"✓ Loaded optimizer state")

        # Optionally override optimizer hyperparameters after loading state
        if cfg.resume_override_lr:
            for g in opt.param_groups:
                g["lr"] = cfg.lr
            print(f"↻ Resume: overriding lr to {cfg.lr}")

        if cfg.resume_override_weight_decay:
            # Keep no_decay group at 0.0; apply cfg.weight_decay to the decay group only.
            if opt.param_groups:
                opt.param_groups[0]["weight_decay"] = cfg.weight_decay
            if len(opt.param_groups) > 1:
                opt.param_groups[1]["weight_decay"] = 0.0
            print(f"↻ Resume: overriding weight_decay to {cfg.weight_decay}")

        # Load / reset scheduler state
        if cfg.resume_reset_scheduler:
            if sched_mode == "cosine":
                # Rebuild a cosine schedule over the remaining epochs.
                if start_epoch <= cfg.warmup_epochs:
                    t_max = max(1, cfg.epochs - cfg.warmup_epochs)
                else:
                    t_max = max(1, cfg.epochs - start_epoch + 1)
                base_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt,
                    T_max=t_max,
                    eta_min=float(getattr(cfg, "min_lr", 0.0)),
                )
                print(f"↻ Resume: resetting scheduler (CosineAnnealingLR, T_max={t_max})")
            elif sched_mode == "onecycle":
                steps_per_epoch = len(train_loader)
                total_steps = max(1, cfg.epochs * steps_per_epoch)
                base_sched = torch.optim.lr_scheduler.OneCycleLR(
                    opt,
                    max_lr=cfg.lr,
                    total_steps=total_steps,
                    pct_start=float(getattr(cfg, "onecycle_pct_start", 0.1)),
                    div_factor=float(getattr(cfg, "onecycle_div_factor", 25.0)),
                    final_div_factor=float(getattr(cfg, "onecycle_final_div_factor", 1e4)),
                    anneal_strategy="cos",
                )
                print("↻ Resume: resetting scheduler (OneCycleLR)")
            else:
                base_sched = None
                print("↻ Resume: scheduler disabled (--scheduler none)")
        else:
            if checkpoint.get("scheduler_state_dict") is not None and base_sched is not None:
                try:
                    base_sched.load_state_dict(checkpoint["scheduler_state_dict"])
                    print("✓ Loaded scheduler state")
                except Exception as e:
                    print(f"⚠️  Failed to load scheduler state ({type(e).__name__}: {e}). Continuing with fresh scheduler.")
        
        # Load history (handle both dict and list formats for backward compatibility)
        saved_history = checkpoint.get('history', None)
        if saved_history:
            if isinstance(saved_history, dict):
                history = saved_history
            elif isinstance(saved_history, list):
                # Convert old list format to new dict format
                for h in saved_history:
                    history['epochs'].append(h['epoch'])
                    history['train_loss'].append(h['train_loss'])
                    history['train_acc'].append(h['train_acc'] * 100)
                    history['test_loss'].append(h['test_loss'])
                    history['test_acc'].append(h['test_acc'] * 100)
        
        print(f"✓ Resuming from epoch {start_epoch}")
        print(f"✓ Best accuracy so far: {best*100:.2f}% (epoch {best_epoch})")
        if history['epochs']:
            print(f"✓ Loaded {len(history['epochs'])} historical epochs")
        
        # Verify config compatibility (warn if different)
        saved_cfg = checkpoint.get('config')
        if saved_cfg is not None:
            config_warnings = []
            critical_params = ['state_dim', 'd_model', 'n_layers', 'use_stacked', 'use_z_gate']
            for param in critical_params:
                if hasattr(saved_cfg, param) and hasattr(cfg, param):
                    if getattr(saved_cfg, param) != getattr(cfg, param):
                        config_warnings.append(f"  ⚠️  {param}: checkpoint={getattr(saved_cfg, param)}, current={getattr(cfg, param)}")
            
            if config_warnings:
                print("\n⚠️  WARNING: Configuration mismatch detected:")
                for warning in config_warnings:
                    print(warning)
                print("Model may not resume correctly with different architecture!\n")
        
        print(f"{'='*60}\n")

    def run_epoch(loader, train=True, epoch=1):
        model.train(train)
        total, correct, total_loss = 0, 0, 0.0
        phase = "Train" if train else "Eval"
        pbar = tqdm(loader, total=len(loader), desc=f"{phase} {epoch}/{cfg.epochs}",
                    dynamic_ncols=False, leave=False, ncols=100, 
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                    position=0, file=sys.stdout)
        cm = torch.enable_grad() if train else torch.no_grad()
        
        # Performance monitoring
        start_time = time.time()
        batch_times = []
        
        with cm:
            for batch_idx, (xb, yb) in enumerate(pbar):
                batch_start = time.time()
                
                x = xb.to(device, non_blocking=True).transpose(0, 1)  # [T,B,1]
                # normalize in-place using dataset stats
                x = (x - cfg.input_mean) / cfg.input_std
                # Add pixel noise augmentation during training
                if train and cfg.pixel_noise > 0:
                    x = x + torch.randn_like(x) * cfg.pixel_noise
                y = yb.to(device, non_blocking=True)
                
                # Reset model state before each sequence
                model.reset_state(x.size(1), x.device)

                amp_cm = torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=(cfg.amp_bf16 and device.type == "cuda"),
                )
                with amp_cm:
                    logits = model(x)
                    loss = F.cross_entropy(logits, y, label_smoothing=cfg.label_smoothing)

                if train:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
                    opt.step()
                    step_sched_batch()

                total += y.size(0)
                correct += (logits.argmax(dim=-1) == y).sum().item()
                total_loss += loss.item() * y.size(0)

                # Performance tracking
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                
                # Compute debug info more frequently for better monitoring
                if batch_idx % 1500 == 0:  # Every 5th batch instead of 50th
                    # Debug peek only available for single-layer model
                    if hasattr(model.core, 'debug_peek'):
                        # snapshot state for safe debug
                        _gin_state = getattr(model.core.gin, "syn_state", None)
                        _dt_state  = getattr(model.core.dt, "syn_state", None)
                        dbg = model.core.debug_peek(x[-1])
                        dbg_info = {
                            'dt': f"{dbg.get('dt', 0.0):.3f}",
                            'gin': f"{dbg.get('gin', 0.0):.3f}",
                            'c_norm': f"{dbg.get('c_norm', 0.0):.2f}",
                            'phi': f"{dbg.get('phi', 0.0):.3f}",
                            'inj': f"{dbg.get('inj', 0.0):.3f}",
                        }
                        # restore states (avoid lingering side-effects)
                        if hasattr(model.core.gin, "syn_state"):
                            model.core.gin.syn_state = _gin_state
                        if hasattr(model.core.dt, "syn_state"):
                            model.core.dt.syn_state = _dt_state
                    else:
                        # Stacked model: use placeholder values
                        dbg_info = {
                            'dt': "stack",
                            'gin': "stack",
                            'c_norm': "stack",
                            'phi': "stack",
                            'inj': "stack",
                        }
                else:
                    # Use previous values instead of dashes
                    dbg_info = {
                        'dt': f"{0.100:.3f}",  # Default values
                        'gin': f"{0.500:.3f}",
                        'c_norm': f"{0.70:.2f}",
                        'phi': f"{0.985:.3f}",
                        'inj': f"{0.034:.3f}",
                    }

                # Calculate throughput
                avg_batch_time = sum(batch_times[-10:]) / min(10, len(batch_times))
                throughput = x.size(1) / avg_batch_time if avg_batch_time > 0 else 0

                # Enhanced progress display - simplified for cleaner logs
                pbar.set_postfix(
                    loss=f"{(total_loss / total):.4f}",
                    acc=f"{(100.0 * correct / total):.2f}%",
                    lr=f"{opt.param_groups[0]['lr']:.2e}",
                    dt=dbg_info['dt'],
                    gin=dbg_info['gin'],
                    speed=f"{throughput:.1f} seq/s"
                )
        
        # Print performance summary
        total_time = time.time() - start_time
        avg_batch_time = sum(batch_times) / len(batch_times)
        final_throughput = total/len(batch_times)/avg_batch_time
        
        # Print epoch summary with proper formatting
        print(f"\n[{phase}] Epoch {epoch} Summary:")
        print(f"  Time: {total_time:.2f}s | Avg batch: {avg_batch_time:.3f}s")
        print(f"  Throughput: {final_throughput:.1f} samples/s | Samples: {total:,}")
        print(f"  Loss: {total_loss/total:.4f} | Accuracy: {100.0*correct/total:.2f}%")
        
        # Memory usage (if CUDA)
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  GPU memory: {memory_allocated:.2f}GB allocated | {memory_reserved:.2f}GB reserved")
        
        # Flush output to ensure it appears in logs
        sys.stdout.flush()
        
        return total_loss / total, correct / total

    # These are now initialized above in the resume section
    # best = 0.0
    # best_epoch = 0
    # patience_counter = 0
    # history = {...} (dict format)
    
    training_start_time = time.time()
    patience = cfg.patience  # Early stopping patience (<=0 disables)
    
    print("\nTraining progress:")
    print(f"{'Epoch':<6} {'Train Loss':<10} {'Train Acc':<10} {'Test Loss':<10} {'Test Acc':<10} {'Best':<8} {'Status'}")
    print(f"{'-'*70}")
    
    for ep in range(start_epoch, cfg.epochs + 1):
        tr_loss, tr_acc = run_epoch(train_loader, True, ep)
        te_loss, te_acc = run_epoch(test_loader, False, ep)
        step_sched_epoch(ep)
        
        # Track metrics
        history['epochs'].append(ep)
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc * 100)  # Convert to percentage
        history['test_loss'].append(te_loss)
        history['test_acc'].append(te_acc * 100)  # Convert to percentage
        
        # Track best performance
        is_best = te_acc > best
        if is_best:
            best = te_acc
            best_epoch = ep
            patience_counter = 0
            
            # Save best model checkpoint
            checkpoint_dir = Path(cfg.checkpoint_dir)
            checkpoint_dir.mkdir(exist_ok=True)
            
            # Create checkpoint with all necessary info
            checkpoint = {
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scheduler_state_dict': base_sched.state_dict() if base_sched is not None else None,
                'best_acc': best,
                'train_acc': tr_acc,
                'train_loss': tr_loss,
                'test_acc': te_acc,
                'test_loss': te_loss,
                'config': cfg.__dict__,
            }
            
            # Save ONLY as "latest_best.pt" (overwrite) to avoid storage blow-ups.
            model_type = "stacked" if cfg.use_stacked else "single"
            zgate_str = "_zgate" if cfg.use_z_gate else ""
            latest_path = checkpoint_dir / f"latest_best_{model_type}{zgate_str}.pt"
            torch.save(checkpoint, latest_path)
            print(f"  → Saved checkpoint: {latest_path} (acc={te_acc*100:.2f}%, ep={ep})")
            
            # Save convergence plot
            plot_path = checkpoint_dir / f"convergence_{model_type}{zgate_str}_ep{ep}.png"
            plot_training_curves(
                history,
                plot_path,
                title=f"{model_type.capitalize()} SSM{' + Z-gate' if cfg.use_z_gate else ''}",
            )
            
        else:
            patience_counter += 1
        
         
        # Status indicator
        if is_best:
            status = "NEW BEST"
        elif patience > 0 and patience_counter >= patience:
            status = "EARLY STOP"
        elif te_acc > 0.8:
            status = "IMPROVING"
        else:
            status = "RUNNING"
        
        # Learning rate info
        current_lr = opt.param_groups[0]['lr']
        
        print(f"{ep:02d}    {tr_loss:.4f}     {tr_acc*100:6.2f}%    {te_loss:.4f}     {te_acc*100:6.2f}%    {best*100:6.2f}%  {status}")
        
        # Early stopping check
        if patience > 0 and patience_counter >= patience and ep >= 5:  # Don't stop too early (minimum 5 epochs)
            print(f"\nEarly stopping triggered! No improvement for {patience} epochs.")
            print(f"   Best accuracy: {best*100:.2f}% (epoch {best_epoch})")
            break
    
    # Final training summary
    total_training_time = time.time() - training_start_time
    print(f"\n{'='*70}")
    print("TRAINING COMPLETED")
    print(f"{'='*70}")
    print(f"Total time: {total_training_time:.2f}s ({total_training_time/60:.1f} min)")
    print(f"Best accuracy: {best*100:.2f}% (epoch {best_epoch})")
    print(f"Final train: {tr_loss:.4f} loss, {tr_acc*100:.2f}% acc")
    print(f"Final test:  {te_loss:.4f} loss, {te_acc*100:.2f}% acc")
    print(f"{'='*70}")
    
    # Save final model checkpoint (even if not best)
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    final_checkpoint = {
        'epoch': ep,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'scheduler_state_dict': base_sched.state_dict() if base_sched is not None else None,
        'best_acc': best,
        'best_epoch': best_epoch,
        'final_train_acc': tr_acc,
        'final_test_acc': te_acc,
        'total_epochs': ep,
        'training_time': total_training_time,
        'config': cfg.__dict__,
    }
    
    model_type = "stacked" if cfg.use_stacked else "single"
    zgate_str = "_zgate" if cfg.use_z_gate else ""
    final_path = checkpoint_dir / f"final_model_{model_type}{zgate_str}_best{best*100:.2f}_ep{ep}.pt"
    torch.save(final_checkpoint, final_path)
    print(f"→ Saved final checkpoint: {final_path}")
    
    # Save final convergence plot
    final_plot_path = checkpoint_dir / f"final_convergence_{model_type}{zgate_str}_ep{ep}.png"
    plot_training_curves(
        history,
        final_plot_path,
        title=f"{model_type.capitalize()} SSM{' + Z-gate' if cfg.use_z_gate else ''} - Final",
    )
    
    return best

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="pMNIST with FULL selective SSM (input-dependent C) using SpikingJelly")
    p.add_argument(
        "--tiny",
        action="store_true",
        help="use a very small selective SSM preset for quick sanity checks",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "onecycle", "none"],
        help="lr schedule: cosine (warmup+cosine), onecycle (per-batch), none (constant)",
    )
    p.add_argument("--min-lr", type=float, default=0.0, help="cosine eta_min (ignored for onecycle/none)")
    p.add_argument("--onecycle-pct-start", type=float, default=0.1, help="OneCycleLR pct_start (ignored unless --scheduler onecycle)")
    p.add_argument("--onecycle-div-factor", type=float, default=25.0, help="OneCycleLR div_factor (ignored unless --scheduler onecycle)")
    p.add_argument("--onecycle-final-div-factor", type=float, default=1e4, help="OneCycleLR final_div_factor (ignored unless --scheduler onecycle)")
    p.add_argument("--patience", type=int, default=5, help="early stopping patience (<=0 disables)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume from")
    p.add_argument("--resume-reset-optimizer", action="store_true", help="on resume: reset optimizer state (for fine-tuning)")
    p.add_argument("--resume-reset-scheduler", action="store_true", help="on resume: reset scheduler state (for fine-tuning)")
    p.add_argument("--resume-override-lr", action="store_true", help="on resume: force --lr onto optimizer after loading state")
    p.add_argument("--resume-override-weight-decay", action="store_true", help="on resume: force --weight-decay onto optimizer after loading state")
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="directory to save checkpoints/plots (use a unique path per run to avoid collisions)",
    )
    p.add_argument("--state-dim", type=int, default=256)
    p.add_argument("--gate-hidden", type=int, default=128)
    p.add_argument("--dt-min", type=float, default=1e-3)
    p.add_argument("--dt-max", type=float, default=0.2)
    p.add_argument("--readout-dim", type=int, default=64)
    p.add_argument("--last-k-avg", type=int, default=32)
    p.add_argument("--head-hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--feat-dim", type=int, default=16, help="feature size for gate inputs")
    p.add_argument("--perm-seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--enable-cudagraphs", action="store_true")
    p.add_argument(
        "--amp-bf16",
        action="store_true",
        help="enable bf16 autocast on CUDA for speed (keeps spiking gate precompute in fp32)",
    )
    p.add_argument("--cache-dir", type=str, default="./data/PMNIST")
    p.add_argument("--rebuild-cache", action="store_true")
    # Gate parameters (spiking-only)
    p.add_argument("--syn-alpha", type=float, default=0.8, help="EMA alpha for spiking gate readouts")
    p.add_argument("--gate-temp", type=float, default=1.0, help="temperature for gate sigmoids (>1 flattens)")
    p.add_argument("--no-input-norm", action="store_true", help="disable dataset mean/std normalization")
    # LIF neuron parameters
    p.add_argument("--lif-tau", type=float, default=2.0, help="LIF membrane time constant")
    p.add_argument("--lif-v-threshold", type=float, default=1.0, help="LIF spike threshold")
    p.add_argument("--lif-v-reset", type=float, default=0.0, help="LIF reset potential")
    # C network
    p.add_argument("--c-hidden", type=int, default=64, help="hidden size for selective C network")
    p.add_argument("--c-rank", type=int, default=None, help="rank for C low-rank decomposition (default: min(128, state_dim//2))")
    # Phase 1: Stacked model (for higher accuracy)
    p.add_argument("--stacked", action="store_true", help="use stacked SSM architecture (Phase 1)")
    p.add_argument("--n-layers", type=int, default=4, help="number of SSM layers (stacked mode)")
    p.add_argument("--d-model", type=int, default=128, help="model dimension (stacked mode)")
    p.add_argument("--conv-kernel", type=int, default=4, help="depthwise conv kernel size (stacked mode)")
    # Phase 2: Z-gate (output gating)
    p.add_argument("--z-gate", action="store_true", help="enable Mamba-style output gating (stacked mode)")
    # Regularization
    p.add_argument("--label-smoothing", type=float, default=0.0, help="label smoothing for cross-entropy (0.1 recommended)")
    p.add_argument("--pixel-noise", type=float, default=0.0, help="Gaussian noise std to add to pixels during training (0.05-0.1 recommended)")
    return p.parse_args()


def apply_tiny_preset(args: argparse.Namespace) -> argparse.Namespace:
    if not args.tiny:
        return args
    args.state_dim = 64
    args.gate_hidden = 32
    args.readout_dim = 32
    args.head_hidden = 32
    args.feat_dim = 8
    args.c_hidden = 16
    args.c_rank = 16
    args.last_k_avg = 8
    args.batch_size = 64
    args.stacked = False
    args.n_layers = 1
    args.d_model = 64
    args.conv_kernel = 3
    args.quick = True
    return args


def main():
    args = apply_tiny_preset(parse_args())
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        clip=args.clip,
        warmup_epochs=args.warmup_epochs,
        scheduler=args.scheduler,
        min_lr=args.min_lr,
        onecycle_pct_start=args.onecycle_pct_start,
        onecycle_div_factor=args.onecycle_div_factor,
        onecycle_final_div_factor=args.onecycle_final_div_factor,
        patience=args.patience,
        seed=args.seed,
        state_dim=args.state_dim,
        gate_hidden=args.gate_hidden,
        dt_min=args.dt_min,
        dt_max=args.dt_max,
        readout_dim=args.readout_dim,
        last_k_avg=args.last_k_avg,
        head_hidden=args.head_hidden,
        dropout=args.dropout,
        feat_dim=args.feat_dim,
        perm_seed=args.perm_seed,
        quick=args.quick,
        enable_cudagraphs=args.enable_cudagraphs,
        amp_bf16=args.amp_bf16,
        cache_dir=args.cache_dir,
        rebuild_cache=args.rebuild_cache,
        resume=args.resume,
        resume_reset_optimizer=args.resume_reset_optimizer,
        resume_reset_scheduler=args.resume_reset_scheduler,
        resume_override_lr=args.resume_override_lr,
        resume_override_weight_decay=args.resume_override_weight_decay,
        checkpoint_dir=args.checkpoint_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
        syn_alpha=args.syn_alpha,
        gate_temp=args.gate_temp,
        input_norm=not args.no_input_norm,
        lif_tau=args.lif_tau,
        lif_v_threshold=args.lif_v_threshold,
        lif_v_reset=args.lif_v_reset,
        c_hidden=args.c_hidden,
        c_rank=args.c_rank,
        # Phase 1: Stacked model
        use_stacked=args.stacked,
        n_layers=args.n_layers,
        d_model=args.d_model,
        conv_kernel=args.conv_kernel,
        # Phase 2: Z-gate
        use_z_gate=args.z_gate,
        # Regularization
        label_smoothing=args.label_smoothing,
        pixel_noise=args.pixel_noise,
    )
    best = train_pmnist(cfg)
    print(f"Best test accuracy: {best * 100:.2f}%")
    
    # Additional useful information
    print("\nAdditional information:")
    if cfg.use_stacked:
        print(f"  - Stacked SSM: {cfg.n_layers} layers, d_model={cfg.d_model}, conv_kernel={cfg.conv_kernel}")
    else:
        print("  - Single-layer SSM")
    print("  - Model uses spiking gates (SpikingJelly LIF)")
    print(f"  - C matrix uses low-rank approximation (rank {cfg.c_rank or min(128, cfg.state_dim//2)})")
    print("  - Vectorized forward pass enabled for speed")
    print(f"  - Input normalization: {'enabled' if cfg.input_norm else 'disabled'}")

if __name__ == "__main__":
    main()

# NOTE:
# For a side-by-side comparison with Mamba-style selective SSMs and a checklist
# for moving this implementation closer to a full Mamba block, see:
#   docs/MAMBA_SELECTIVE_COMPARISON.md