"""
Phase 5.1 — Open-loop live inference (predict-but-don't-act).

Streams DAVIS346 events, bins them into 10-ms ticks, runs the v3 model
and overlays predictions on a live display. Nothing is sent to the arm.

Display layout (side by side):
  LEFT  — 260×260 event accumulation
            cyan   circle  = v3 prediction (t + 50 ms)
            yellow dot     = current event centroid
            magenta dot    = centroid_now baseline
  RIGHT — APS frame with the same overlays

Bottom HUD: tick, fps, inference ms, n_events, pred xy, centroid xy,
            signed lead (positive = model predicts AHEAD of centroid)

Log (CSV):  phase51_log_<timestamp>.csv  — one row per 10-ms tick

Usage:
    python phase51_open_loop.py [--no-preview]

Keys:
    Q / ESC  quit
    R        reset model state
    S        save current frame
"""

import argparse
import csv
import os
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
import torch

# ── paths ─────────────────────────────────────────────────────────────────────
HERE     = os.path.dirname(os.path.abspath(__file__))
PRED_DIR = os.path.join(HERE, 'v3_predictor_release')
sys.path.insert(0, PRED_DIR)

import dv_processing as dv
from model_loader import build_from_args

# ── constants ──────────────────────────────────────────────────────────────────
DAVIS_W, DAVIS_H = 346, 260
CROP_W            = 260                        # centre-crop to 260×260
CROP_X0           = (DAVIS_W - CROP_W) // 2   # = 43
BIN_MS            = 10.0                       # event accumulation window
DISPLAY_SCALE     = 2                          # upscale for visibility
CHECKPOINT        = os.path.join(PRED_DIR, 'best.pt')
LOG_DIR           = os.path.join(HERE, 'phase51_logs')

# Background Activity Filter window (µs) — Inivation's official BAF
BAF_WINDOW_US     = 10_000   # 10 ms; increase if too much noise

# ── event ring buffer shared between camera and inference threads ─────────────

class EventBuffer:
    def __init__(self, resolution, baf_us=BAF_WINDOW_US):
        self._x    = deque()
        self._y    = deque()
        self._t    = deque()
        self._p    = deque()
        self._lock = threading.Lock()
        import datetime
        self._baf  = dv.noise.BackgroundActivityNoiseFilter(
            resolution,
            backgroundActivityDuration=datetime.timedelta(microseconds=baf_us))

    def push(self, batch):
        # Run BAF (filters noise events with no recent spatial neighbour)
        self._baf.accept(batch)
        filtered = self._baf.generateEvents()
        if filtered is None or len(filtered) == 0:
            return

        coords = np.array(filtered.coordinates(), dtype=np.int32)
        pols   = np.array(filtered.polarities(),  dtype=np.bool_)
        ts     = np.array(filtered.timestamps(),  dtype=np.int64)

        # Centre-crop: keep only x in [CROP_X0, CROP_X0+CROP_W)
        mask = (coords[:, 0] >= CROP_X0) & (coords[:, 0] < CROP_X0 + CROP_W)
        if not mask.any():
            return
        x = (coords[mask, 0] - CROP_X0).astype(np.int16)
        y = coords[mask, 1].astype(np.int16)
        t = ts[mask]
        p = pols[mask]

        with self._lock:
            self._x.extend(x.tolist())
            self._y.extend(y.tolist())
            self._t.extend(t.tolist())
            self._p.extend(p.tolist())

    def drain_before(self, t_us):
        """Return all events with timestamp < t_us as numpy arrays."""
        with self._lock:
            out_x, out_y, out_t, out_p = [], [], [], []
            while self._t and self._t[0] < t_us:
                out_x.append(self._x.popleft())
                out_y.append(self._y.popleft())
                out_t.append(self._t.popleft())
                out_p.append(self._p.popleft())
        if not out_t:
            return None
        return (np.array(out_x, np.int32), np.array(out_y, np.int32),
                np.array(out_t, np.int64), np.array(out_p, bool))

    def latest_ts(self):
        with self._lock:
            return self._t[-1] if self._t else None


# ── binning ────────────────────────────────────────────────────────────────────

def bin_events(x, y, p, hw=(DAVIS_H, CROP_W)):
    """Single 10-ms bin → (H, W, 2) float32."""
    H, W = hw
    b = np.zeros((H, W, 2), dtype=np.float32)
    if x is None or len(x) == 0:
        return b
    xc = np.clip(x, 0, W - 1)
    yc = np.clip(y, 0, H - 1)
    np.add.at(b, (yc[p],  xc[p],  0), 1.0)
    np.add.at(b, (yc[~p], xc[~p], 1), 1.0)
    return b


def event_centroid(x, y):
    """(cx, cy) of event cloud, or None."""
    if x is None or len(x) == 0:
        return None
    return float(np.mean(x)), float(np.mean(y))


# ── visualisation ──────────────────────────────────────────────────────────────

def make_event_img(bin_hw):
    """Render a binned event array (H,W,2) as BGR."""
    on  = bin_hw[:, :, 0]
    off = bin_hw[:, :, 1]
    img = np.full((*on.shape, 3), 30, dtype=np.uint8)
    img[on  > 0] = (0, 220, 0)
    img[off > 0] = (0, 0, 220)
    return img


def draw_overlays(img, pred_pix, centroid, scale=1):
    s = scale
    if pred_pix is not None:
        cx, cy = int(pred_pix[0] * s), int(pred_pix[1] * s)
        cv2.circle(img, (cx, cy), 10 * s, (255, 220, 0), 2)
        cv2.drawMarker(img, (cx, cy), (255, 220, 0),
                       cv2.MARKER_CROSS, 16 * s, 2)
    if centroid is not None:
        cx, cy = int(centroid[0] * s), int(centroid[1] * s)
        cv2.circle(img, (cx, cy), 5 * s, (0, 255, 255), -1)


def hud_text(img, lines, start_y=14, lh=18, color=(200, 200, 200)):
    for i, line in enumerate(lines):
        cv2.putText(img, line, (6, start_y + i * lh),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-preview', action='store_true')
    parser.add_argument('--baf', type=int, default=BAF_WINDOW_US,
                        help='Background Activity Filter window in µs (default 10000)')
    args = parser.parse_args()
    preview = not args.no_preview

    # ── load model ────────────────────────────────────────────────────────────
    print(f'Loading {CHECKPOINT}')
    ckpt   = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = build_from_args(ckpt['args'], state_dict=ckpt['model_state_dict']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    model.reset_state(1, device=device)
    # Warm up CUDA so first live tick doesn't stall
    _dummy = torch.zeros(1, 1, 260, 260, 2, device=device)
    with torch.no_grad():
        for _ in range(5):
            model(_dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    model.reset_state(1, device=device)
    print(f'Model ready on {device}  (epoch={ckpt.get("epoch")}, '
          f'val_rmse={ckpt.get("val_rmse_px"):.2f} px)')

    # ── open camera ───────────────────────────────────────────────────────────
    devices = dv.io.camera.discover()
    if not devices:
        print('ERROR: No DAVIS camera found.'); return
    cam = dv.io.camera.open(devices[0].serialNumber)
    print(f'Camera: {cam.getCameraName()}  {cam.getEventResolution()}')

    buf = EventBuffer(cam.getEventResolution(), baf_us=args.baf)
    print(f'BAF window: {args.baf} µs')
    last_aps = np.full((DAVIS_H, DAVIS_W), 64, dtype=np.uint8)

    def cam_thread_fn():
        while cam_running[0]:
            b = cam.getNextEventBatch()
            if b and len(b) > 0:
                buf.push(b)
            f = cam.getNextFrame()
            if f is not None:
                last_aps[:] = f.image if f.image.ndim == 2 else f.image[:, :, 0]

    cam_running = [True]
    ct = threading.Thread(target=cam_thread_fn, daemon=True)
    ct.start()
    time.sleep(0.5)   # let buffer fill

    # ── logging ───────────────────────────────────────────────────────────────
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f'phase51_{time.strftime("%Y%m%d_%H%M%S")}.csv')
    log_fields = ['tick', 'wall_s', 'pred_cx', 'pred_cy',
                  'centroid_cx', 'centroid_cy', 'n_events',
                  'infer_ms', 'lead_x', 'lead_y']
    log_f = open(log_path, 'w', newline='')
    log_w = csv.DictWriter(log_f, fieldnames=log_fields)
    log_w.writeheader()
    print(f'Logging to {log_path}')

    if preview:
        W_disp = CROP_W * DISPLAY_SCALE
        H_disp = DAVIS_H * DISPLAY_SCALE
        win_w  = W_disp * 2 + 6
        cv2.namedWindow('Phase 5.1 — Live Inference', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Phase 5.1 — Live Inference', win_w, H_disp + 60)

    # ── tick loop ─────────────────────────────────────────────────────────────
    tick       = 0
    t_wall0    = time.time()
    t_mono0    = time.monotonic()
    fps_buf    = deque(maxlen=20)
    pred_pix   = None
    centroid   = None

    print('Running — press Q in window to quit.\n')

    try:
        while True:
            # Wall-clock timing drives ticks (robust when events are sparse)
            next_tick_wall = t_mono0 + (tick + 1) * BIN_MS / 1000.0
            sleep_s = next_tick_wall - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)

            t_infer_start = time.time()

            # Drain all events buffered so far
            tick_end_us = int(time.time() * 1e6)
            evs = buf.drain_before(tick_end_us)
            if evs is not None:
                x, y, t_ev, p = evs
                n_ev     = len(x)
                binned   = bin_events(x, y, p)
                centroid = event_centroid(x, y)
            else:
                n_ev     = 0
                binned   = np.zeros((DAVIS_H, CROP_W, 2), dtype=np.float32)
                centroid = None

            # Model inference (T=1 streaming)
            bin_t = torch.from_numpy(binned).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_norm = model(bin_t)               # (1,1,2)
                if pred_norm.dim() == 4:
                    pred_norm = pred_norm[:, :, 0, :]
                pred_pix = model.norm_to_pix(pred_norm).squeeze().cpu().numpy()  # (2,)

            infer_ms = (time.time() - t_infer_start) * 1000
            fps_buf.append(1.0 / max((time.time() - t_infer_start + BIN_MS * 1e-3), 1e-6))

            # Signed lead: how far ahead of centroid is the prediction?
            lead_x = float(pred_pix[0] - centroid[0]) if centroid else 0.0
            lead_y = float(pred_pix[1] - centroid[1]) if centroid else 0.0

            # Log
            log_w.writerow({
                'tick':        tick,
                'wall_s':      round(time.monotonic() - t_mono0, 4),
                'pred_cx':     round(float(pred_pix[0]), 2),
                'pred_cy':     round(float(pred_pix[1]), 2),
                'centroid_cx': round(centroid[0], 2) if centroid else '',
                'centroid_cy': round(centroid[1], 2) if centroid else '',
                'n_events':    n_ev,
                'infer_ms':    round(infer_ms, 2),
                'lead_x':      round(lead_x, 2),
                'lead_y':      round(lead_y, 2),
            })
            log_f.flush()

            # Console summary every 50 ticks
            if tick % 50 == 0:
                fps_now = 1000.0 / BIN_MS   # theoretical max
                print(f'tick={tick:5d}  pred=({pred_pix[0]:6.1f},{pred_pix[1]:6.1f})'
                      f'  centroid={f"({centroid[0]:.1f},{centroid[1]:.1f})" if centroid else "  none  "}'
                      f'  lead=({lead_x:+.1f},{lead_y:+.1f})'
                      f'  n_ev={n_ev:6d}  infer={infer_ms:.1f}ms')

            # Display
            if preview:
                S = DISPLAY_SCALE
                evt_img  = cv2.resize(make_event_img(binned),
                                      (CROP_W * S, DAVIS_H * S),
                                      interpolation=cv2.INTER_NEAREST)
                aps_crop = last_aps[:, CROP_X0:CROP_X0 + CROP_W]
                aps_img  = cv2.resize(cv2.cvtColor(aps_crop, cv2.COLOR_GRAY2BGR),
                                      (CROP_W * S, DAVIS_H * S),
                                      interpolation=cv2.INTER_LINEAR)

                draw_overlays(evt_img, pred_pix, centroid, scale=S)
                draw_overlays(aps_img, pred_pix, centroid, scale=S)

                # HUD
                fps_est = len(fps_buf) / max(sum(1/f for f in fps_buf if f > 0), 1e-9)
                hud_lines_l = [
                    f'tick {tick}   {BIN_MS:.0f}ms bins',
                    f'events: {n_ev}',
                    f'infer:  {infer_ms:.1f} ms',
                ]
                hud_lines_r = [
                    f'pred  ({pred_pix[0]:6.1f}, {pred_pix[1]:6.1f})',
                    f'cent  ({centroid[0]:6.1f}, {centroid[1]:6.1f})' if centroid else 'cent  --',
                    f'lead  ({lead_x:+.1f}, {lead_y:+.1f})',
                ]
                hud_text(evt_img, hud_lines_l)
                hud_text(aps_img, hud_lines_r)

                # Legend
                cv2.circle(evt_img,  (CROP_W*S - 20, DAVIS_H*S - 30), 8*S//2, (255,220,0), 2)
                cv2.putText(evt_img, 'pred+50ms', (CROP_W*S-110, DAVIS_H*S-26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,220,0), 1)
                cv2.circle(evt_img,  (CROP_W*S - 20, DAVIS_H*S - 12), 4*S//2, (0,255,255), -1)
                cv2.putText(evt_img, 'centroid',  (CROP_W*S-110, DAVIS_H*S-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,255,255), 1)

                div     = np.full((DAVIS_H * S, 6, 3), 40, dtype=np.uint8)
                combined = np.hstack([evt_img, div, aps_img])
                cv2.imshow('Phase 5.1 — Live Inference', combined)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break
                elif key in (ord('r'), ord('R')):
                    model.reset_state(1, device=device)
                    print('Model state reset.')
                elif key in (ord('s'), ord('S')):
                    fname = os.path.join(LOG_DIR, f'snap_{tick}.png')
                    cv2.imwrite(fname, combined)
                    print(f'Saved {fname}')

            tick += 1

    except KeyboardInterrupt:
        print('\nInterrupted.')
    finally:
        cam_running[0] = False
        log_f.close()
        if preview:
            cv2.destroyAllWindows()
        print(f'\nLog saved → {log_path}')
        print(f'Total ticks: {tick}  ({tick * BIN_MS / 1000:.1f} s)')


if __name__ == '__main__':
    main()
