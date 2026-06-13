"""Stateful Python event simulator for closed-loop use.

Matches ESIM's core algorithm (log-intensity differencing per pixel) with the
same parameter set: Cp, Cn, refractory, log_eps, use_log. NOT a byte-for-byte
replacement — it skips ESIM's within-frame linear interpolation of event
timestamps. For our 10-ms binning, this is irrelevant: every event from a given
frame lands in the same bin regardless of sub-frame timestamping.

API:

    sim = EventSim((H, W), cp=0.2, cn=0.2, refractory_s=1e-4,
                   log_eps=1e-3, use_log=True)
    sim.reset(first_frame_gray, t=0.0)
    events_a = sim.step(frame_at_t1, t1)
    events_b = sim.step(frame_at_t2, t2)
    ...

Each `step` returns an (N, 4) float32 array [x, y, t_s, polarity] for events
emitted between the previous call and this one. Multiple polarity-aligned
events per pixel are stacked (matches ESIM behaviour: floor(|delta|/C)
events on a large log-intensity step).
"""
from __future__ import annotations

import numpy as np


class EventSim:
    def __init__(self, image_hw: tuple[int, int], cp: float = 0.2, cn: float = 0.2,
                 refractory_s: float = 1e-4, log_eps: float = 1e-3,
                 use_log: bool = True):
        self.H, self.W = image_hw
        self.cp = float(cp)
        self.cn = float(cn)
        self.refractory_s = float(refractory_s)
        self.log_eps = float(log_eps)
        self.use_log = bool(use_log)
        self._ref = None       # (H, W) float64 — last "reference" log-intensity
        self._t_last = None    # (H, W) float64 — last event time per pixel
        self._t_prev = None    # float — last frame's timestamp (for any future interpolation)

    def _map(self, frame_gray: np.ndarray) -> np.ndarray:
        """Map uint8 image to ESIM's working scale (log-intensity if use_log)."""
        x = frame_gray.astype(np.float64) / 255.0
        if self.use_log:
            return np.log(x + self.log_eps)
        return x

    def reset(self, frame_gray: np.ndarray, t: float = 0.0) -> None:
        assert frame_gray.shape == (self.H, self.W)
        self._ref = self._map(frame_gray)
        self._t_last = np.full((self.H, self.W), -np.inf, dtype=np.float64)
        self._t_prev = float(t)

    def step(self, frame_gray: np.ndarray, t: float) -> np.ndarray:
        """Return events emitted between the previous call and this one.

        Events: (N, 4) float32 [x_pix, y_pix, t_s, polarity ∈ {-1, +1}].
        """
        assert self._ref is not None, "Call reset(...) first."
        L = self._map(frame_gray)
        delta = L - self._ref

        # Positive: ON events where delta > Cp (and refractory passed).
        active_pos = (delta > self.cp) & ((t - self._t_last) >= self.refractory_s)
        n_pos = np.zeros_like(delta, dtype=np.int64)
        n_pos[active_pos] = np.floor(delta[active_pos] / self.cp).astype(np.int64)

        # Negative: OFF events where delta < -Cn (refractory).
        active_neg = (delta < -self.cn) & ((t - self._t_last) >= self.refractory_s)
        n_neg = np.zeros_like(delta, dtype=np.int64)
        n_neg[active_neg] = np.floor((-delta[active_neg]) / self.cn).astype(np.int64)

        # Update reference + last-event time wherever events fired.
        any_fired = (n_pos > 0) | (n_neg > 0)
        # Snap reference to ref + n * sign(delta) * C (the residual stays below C).
        self._ref += n_pos * self.cp
        self._ref -= n_neg * self.cn
        self._t_last[any_fired] = t

        # Build the event array.
        total = int(n_pos.sum() + n_neg.sum())
        if total == 0:
            self._t_prev = float(t)
            return np.zeros((0, 4), dtype=np.float32)
        events = np.empty((total, 4), dtype=np.float32)
        # ON events first.
        if n_pos.sum() > 0:
            ys, xs = np.nonzero(n_pos)
            counts = n_pos[ys, xs]
            xs_rep = np.repeat(xs, counts)
            ys_rep = np.repeat(ys, counts)
            events[:xs_rep.size, 0] = xs_rep
            events[:xs_rep.size, 1] = ys_rep
            events[:xs_rep.size, 2] = t
            events[:xs_rep.size, 3] = 1.0
            off_off = xs_rep.size
        else:
            off_off = 0
        if n_neg.sum() > 0:
            ys, xs = np.nonzero(n_neg)
            counts = n_neg[ys, xs]
            xs_rep = np.repeat(xs, counts)
            ys_rep = np.repeat(ys, counts)
            events[off_off:off_off + xs_rep.size, 0] = xs_rep
            events[off_off:off_off + xs_rep.size, 1] = ys_rep
            events[off_off:off_off + xs_rep.size, 2] = t
            events[off_off:off_off + xs_rep.size, 3] = -1.0
        self._t_prev = float(t)
        return events
