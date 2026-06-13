"""
Phase 5.3 — Closed-loop visual servoing (stationary ball).

Control law (per 10-ms tick):
    err_x = pred_cx - IMAGE_CX
    err_y = pred_cy - IMAGE_CY
    Δq_waist    = clip(-K * err_x / dcx_per_waist,    ±MAX_STEP)
    Δq_shoulder = clip(-K * err_y / dcy_per_shoulder, ±MAX_STEP)
    send new absolute position to servos

Display:
    LEFT  — events  (cyan=pred, yellow=centroid, red arrow=error)
    RIGHT — APS frame with same overlays
    HUD   — error, prediction, joint positions, tick rate

Log: phase53_logs/servo_<timestamp>.csv

Usage:
    python phase53_servo.py [--gain 0.1] [--max-step 0.04] [--device /dev/ttyACM0]

Keys:
    Q / ESC   quit (arm returns to HOME)
    SPACE     pause/resume arm motion (camera keeps running)
    R         reset model state
    S         save snapshot
"""

import argparse
import csv
import datetime
import json
import math
import os
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
import torch
import dv_processing as dv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Adaptive_arm_control'))
import dynamixel_sdk as dxl_sdk

PRED_DIR = os.path.join(os.path.dirname(__file__), 'v3_predictor_release')
sys.path.insert(0, PRED_DIR)
from model_loader import build_from_args

# ── constants ──────────────────────────────────────────────────────────────────
DAVIS_W, DAVIS_H  = 346, 260
CROP_W            = 260
CROP_X0           = (DAVIS_W - CROP_W) // 2
IMAGE_CX          = CROP_W  / 2.0   # 130
IMAGE_CY          = DAVIS_H / 2.0   # 130
BAF_US            = 10_000   # 10 ms; increase if too much noise
BIN_MS            = 10.0
DISPLAY_SCALE     = 2
LOG_DIR           = os.path.join(os.path.dirname(__file__), 'phase53_logs')
CHECKPOINT        = os.path.join(PRED_DIR, 'best.pt')

BAUDRATE          = 1_000_000
ADDR_TORQUE       = 64
ADDR_PROF_ACC     = 108
ADDR_PROF_VEL     = 112
ADDR_GOAL_POS     = 116
TICKS_TO_RAD      = 2 * math.pi / 4096
RAD_TO_TICKS      = 4096 / (2 * math.pi)

HOME_TICKS        = {1: 4090, 2: 736, 4: 948, 5: 1645, 6: 2056}
WAIST_ID          = 1
SHOULDER_ID       = 2

# Joint limits in ticks (absolute)
LIMITS = {1: (2363, 5818), 2: (70, 1500)}


# ── event buffer ───────────────────────────────────────────────────────────────
class EventBuffer:
    def __init__(self, resolution, baf_us=BAF_US, roi_bottom=DAVIS_H):
        self._x = deque(); self._y = deque()
        self._t = deque(); self._p = deque()
        self._lock = threading.Lock()
        self._roi_bottom = roi_bottom
        self._baf  = dv.noise.BackgroundActivityNoiseFilter(
            resolution,
            backgroundActivityDuration=datetime.timedelta(microseconds=baf_us))

    def push(self, batch):
        self._baf.accept(batch)
        f = self._baf.generateEvents()
        if f is None or len(f) == 0:
            return
        coords = np.array(f.coordinates(), dtype=np.int32)
        pols   = np.array(f.polarities(),  dtype=np.bool_)
        mask   = (coords[:,0] >= CROP_X0) & (coords[:,0] < CROP_X0 + CROP_W)
        mask  &= (coords[:,1] < self._roi_bottom)
        if not mask.any():
            return
        x = (coords[mask,0] - CROP_X0).astype(np.int16)
        y = coords[mask,1].astype(np.int16)
        with self._lock:
            self._x.extend(x.tolist()); self._y.extend(y.tolist())
            self._t.extend(np.array(f.timestamps(),dtype=np.int64)[mask].tolist())
            self._p.extend(pols[mask].tolist())

    def drain(self):
        with self._lock:
            if not self._x:
                return None
            x = np.array(self._x, np.int32); y = np.array(self._y, np.int32)
            p = np.array(self._p, bool)
            self._x.clear(); self._y.clear()
            self._t.clear(); self._p.clear()
        return x, y, p


# ── binning & centroid ─────────────────────────────────────────────────────────
def bin_events(x, y, p):
    b = np.zeros((DAVIS_H, CROP_W, 2), dtype=np.float32)
    if x is None: return b
    xc = np.clip(x, 0, CROP_W-1); yc = np.clip(y, 0, DAVIS_H-1)
    np.add.at(b, (yc[p],  xc[p],  0), 1.0)
    np.add.at(b, (yc[~p], xc[~p], 1), 1.0)
    return b

def centroid(x, y):
    if x is None or len(x) == 0: return None
    return float(np.mean(x)), float(np.mean(y))

def event_spread(x, y):
    """Return (std_x, std_y) of event cloud. Small = concentrated = ball present."""
    if x is None or len(x) < 10: return None
    return float(np.std(x)), float(np.std(y))


# ── arm driver ─────────────────────────────────────────────────────────────────
class ArmDriver:
    def __init__(self, device):
        self.ph = dxl_sdk.PortHandler(device)
        self.pk = dxl_sdk.PacketHandler(2.0)
        self.ph.openPort(); self.ph.setBaudRate(BAUDRATE); self.ph.clearPort()
        time.sleep(0.2)
        # Reboot + enable
        for ID in HOME_TICKS:
            self.pk.reboot(self.ph, ID); time.sleep(0.12)
        time.sleep(1.5); self.ph.clearPort()
        for ID in HOME_TICKS:
            self.pk.write4ByteTxOnly(self.ph, ID, ADDR_PROF_ACC, 5);  time.sleep(0.05)
            self.pk.write4ByteTxOnly(self.ph, ID, ADDR_PROF_VEL, 15); time.sleep(0.05)
            self.pk.write1ByteTxOnly(self.ph, ID, ADDR_TORQUE,   1);  time.sleep(0.05)
        # Current position in ticks (command-space tracking)
        self.q_ticks = dict(HOME_TICKS)
        print('Arm: ready')

    def goto_home(self):
        for ID, ticks in HOME_TICKS.items():
            self.pk.write4ByteTxOnly(self.ph, ID, ADDR_GOAL_POS,
                                     int(ticks) & 0xFFFFFFFF); time.sleep(0.05)
        self.q_ticks = dict(HOME_TICKS)
        time.sleep(2.0)

    def step(self, dq_waist_rad, dq_shoulder_rad):
        """Apply incremental position commands."""
        dw = int(round(dq_waist_rad    * RAD_TO_TICKS))
        ds = int(round(dq_shoulder_rad * RAD_TO_TICKS))

        new_w = int(np.clip(self.q_ticks[WAIST_ID]    + dw,
                            *LIMITS[WAIST_ID]))
        new_s = int(np.clip(self.q_ticks[SHOULDER_ID] + ds,
                            *LIMITS[SHOULDER_ID]))

        self.pk.write4ByteTxOnly(self.ph, WAIST_ID,    ADDR_GOAL_POS,
                                 new_w & 0xFFFFFFFF); time.sleep(0.03)
        self.pk.write4ByteTxOnly(self.ph, SHOULDER_ID, ADDR_GOAL_POS,
                                 new_s & 0xFFFFFFFF); time.sleep(0.03)

        self.q_ticks[WAIST_ID]    = new_w
        self.q_ticks[SHOULDER_ID] = new_s
        return new_w, new_s

    def disable(self):
        for ID in HOME_TICKS:
            self.pk.write1ByteTxOnly(self.ph, ID, ADDR_TORQUE, 0); time.sleep(0.05)
        self.ph.closePort()


# ── display helpers ────────────────────────────────────────────────────────────
def make_event_img(binned):
    """Render accumulated event bin as BGR — bright=ON, dark=OFF, grey=nothing."""
    on  = np.clip(binned[:,:,0] * 40, 0, 255).astype(np.uint8)
    off = np.clip(binned[:,:,1] * 40, 0, 255).astype(np.uint8)
    img = np.full((*binned.shape[:2], 3), 30, dtype=np.uint8)
    img[:,:,1] = on    # green channel = ON events
    img[:,:,2] = off   # red channel   = OFF events
    return img

def draw_overlays(img, pred, cent, err_x, err_y, s=1):
    cx_img, cy_img = int(IMAGE_CX*s), int(IMAGE_CY*s)
    cv2.drawMarker(img, (cx_img, cy_img), (80,80,80),
                   cv2.MARKER_CROSS, 20*s, 1)
    if pred is not None:
        px, py = int(pred[0]*s), int(pred[1]*s)
        cv2.circle(img, (px, py), 10*s, (255,220,0), 2)
        cv2.drawMarker(img, (px,py), (255,220,0), cv2.MARKER_CROSS, 14*s, 2)
        cv2.arrowedLine(img, (cx_img,cy_img), (px,py), (0,100,255), 2*s,
                        tipLength=0.2)
    if cent is not None:
        cv2.circle(img, (int(cent[0]*s), int(cent[1]*s)), 5*s, (0,255,255), -1)

def hud(img, lines, y0=14, lh=18, col=(200,200,200)):
    for i,l in enumerate(lines):
        cv2.putText(img, l, (6, y0+i*lh),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1, cv2.LINE_AA)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',   default='/dev/ttyACM0')
    parser.add_argument('--gain',     type=float, default=0.1)
    parser.add_argument('--max-step', type=float, default=0.04,
                        help='Max Δq per tick in radians (default 0.04 ≈ 2.3°)')
    parser.add_argument('--no-preview', action='store_true')
    parser.add_argument('--baf', type=int, default=BAF_US,
                        help='BAF window in µs (default 10000)')
    parser.add_argument('--min-events', type=int, default=50,
                        help='Min events per tick to enable arm motion (default 50)')
    parser.add_argument('--max-spread', type=float, default=30.0,
                        help='Max event std-dev in px to confirm target (default 30)')
    parser.add_argument('--roi-bottom', type=int, default=DAVIS_H,
                        help='Ignore events below this y pixel (default=260, full frame)')
    parser.add_argument('--no-home', action='store_true',
                        help='Skip goto_home() on startup — arm stays wherever it is')
    parser.add_argument('--use-centroid', action='store_true',
                        help='Use event centroid directly instead of model prediction')
    args = parser.parse_args()

    # ── calibration ───────────────────────────────────────────────────────────
    calib_path = os.path.join(os.path.dirname(__file__), 'calib.json')
    if not os.path.exists(calib_path):
        print('ERROR: calib.json not found — run phase52_calibrate.py first.')
        return
    calib = json.load(open(calib_path))
    dcx_per_waist    = calib['dcx_per_waist']
    dcy_per_shoulder = calib['dcy_per_shoulder']
    print(f'Calibration loaded:  dcx/waist={dcx_per_waist:.1f}  '
          f'dcy/shoulder={dcy_per_shoulder:.1f}')

    # ── model ─────────────────────────────────────────────────────────────────
    ckpt   = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = build_from_args(ckpt['args'], state_dict=ckpt['model_state_dict']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    _dummy = torch.zeros(1,1,260,260,2,device=device)
    with torch.no_grad():
        for _ in range(5): model(_dummy)
    if device.type=='cuda': torch.cuda.synchronize()
    model.reset_state(1, device=device)
    print(f'Model ready on {device}')

    # ── camera ────────────────────────────────────────────────────────────────
    devices = dv.io.camera.discover()
    if not devices:
        print('ERROR: No DAVIS camera found.'); return
    cam      = dv.io.camera.open(devices[0].serialNumber)
    buf      = EventBuffer(cam.getEventResolution(), baf_us=args.baf,
                          roi_bottom=args.roi_bottom)
    print(f'BAF window: {args.baf} µs   ROI y < {args.roi_bottom}')
    last_aps = np.full((DAVIS_H, DAVIS_W), 64, dtype=np.uint8)

    cam_running = [True]
    def cam_loop():
        while cam_running[0]:
            b = cam.getNextEventBatch()
            if b and len(b) > 0: buf.push(b)
            f = cam.getNextFrame()
            if f is not None:
                last_aps[:] = f.image if f.image.ndim==2 else f.image[:,:,0]
    threading.Thread(target=cam_loop, daemon=True).start()
    time.sleep(0.5)

    # ── arm ───────────────────────────────────────────────────────────────────
    arm = ArmDriver(args.device)
    if not args.no_home:
        arm.goto_home()

    # ── logging ───────────────────────────────────────────────────────────────
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f'servo_{time.strftime("%Y%m%d_%H%M%S")}.csv')
    fields   = ['tick','wall_s','pred_cx','pred_cy','err_x','err_y',
                'dq_waist_rad','dq_shoulder_rad','q_waist_ticks','q_shoulder_ticks',
                'centroid_cx','centroid_cy','n_events','infer_ms']
    log_f = open(log_path, 'w', newline='')
    log_w = csv.DictWriter(log_f, fieldnames=fields)
    log_w.writeheader()
    print(f'Logging → {log_path}')

    preview = not args.no_preview
    if preview:
        S = DISPLAY_SCALE
        cv2.namedWindow('Phase 5.3 — Visual Servo', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Phase 5.3 — Visual Servo', CROP_W*S*2+6, DAVIS_H*S+60)

    # ── async inference thread ────────────────────────────────────────────────
    # Runs model continuously; main loop reads latest result without blocking.
    _infer_lock  = threading.Lock()
    _latest_pred = [np.array([IMAGE_CX, IMAGE_CY], dtype=np.float32)]
    _latest_bin  = [np.zeros((DAVIS_H, CROP_W, 2), dtype=np.float32)]
    _infer_ms    = [0.0]
    _infer_running = [True]

    def infer_loop():
        while _infer_running[0]:
            with _infer_lock:
                b = _latest_bin[0].copy()
            bin_t = torch.from_numpy(b).unsqueeze(0).unsqueeze(0).to(device)
            t_inf = time.time()
            with torch.no_grad():
                pn = model(bin_t)
                if pn.dim() == 4: pn = pn[:, :, 0, :]
                p = model.norm_to_pix(pn).squeeze().cpu().numpy()
            with _infer_lock:
                _latest_pred[0] = p
                _infer_ms[0]    = (time.time() - t_inf) * 1000

    infer_thread = threading.Thread(target=infer_loop, daemon=True)
    infer_thread.start()

    # ── control loop ──────────────────────────────────────────────────────────
    tick    = 0
    paused  = True
    t_mono0 = time.monotonic()
    pred_pix = np.array([IMAGE_CX, IMAGE_CY], dtype=np.float32)
    cent     = None

    print('\nRunning — SPACE to pause, Q to quit.\n')

    try:
        while True:
            next_tick = t_mono0 + (tick+1) * BIN_MS / 1000.0
            sleep_s   = next_tick - time.monotonic()
            if sleep_s > 0: time.sleep(sleep_s)

            t0 = time.time()

            # Drain events and push latest bin to inference thread
            evs    = buf.drain()
            n_ev   = len(evs[0]) if evs else 0
            binned = bin_events(*evs) if evs else np.zeros((DAVIS_H,CROP_W,2),dtype=np.float32)
            cent   = centroid(evs[0], evs[1]) if evs else None
            with _infer_lock:
                _latest_bin[0] = binned
                pred_pix       = _latest_pred[0].copy()

            infer_ms = _infer_ms[0]

            # Control — target visible only if events are spatially concentrated
            spread = event_spread(evs[0], evs[1]) if evs else None
            target_visible = (
                n_ev >= args.min_events and
                spread is not None and
                spread[0] < args.max_spread and
                spread[1] < args.max_spread
            )

            # Error source: centroid (direct) or model prediction
            if args.use_centroid and cent is not None:
                ctrl_x, ctrl_y = cent[0], cent[1]
            else:
                ctrl_x, ctrl_y = float(pred_pix[0]), float(pred_pix[1])
            err_x = ctrl_x - IMAGE_CX
            err_y = ctrl_y - IMAGE_CY

            dq_w = float(np.clip(-args.gain * err_x / dcx_per_waist,
                                 -args.max_step, args.max_step))
            dq_s = float(np.clip(-args.gain * err_y / dcy_per_shoulder,
                                 -args.max_step, args.max_step))

            if not paused and target_visible:
                q_w, q_s = arm.step(dq_w, dq_s)
            else:
                q_w = arm.q_ticks[WAIST_ID]
                q_s = arm.q_ticks[SHOULDER_ID]
                dq_w = dq_s = 0.0

            # Log
            log_w.writerow({
                'tick': tick,
                'wall_s': round(time.monotonic()-t_mono0, 4),
                'pred_cx': round(float(pred_pix[0]),2),
                'pred_cy': round(float(pred_pix[1]),2),
                'err_x':   round(err_x,2),
                'err_y':   round(err_y,2),
                'dq_waist_rad':    round(dq_w,5),
                'dq_shoulder_rad': round(dq_s,5),
                'q_waist_ticks':    q_w,
                'q_shoulder_ticks': q_s,
                'centroid_cx': round(cent[0],2) if cent else '',
                'centroid_cy': round(cent[1],2) if cent else '',
                'n_events':  n_ev,
                'infer_ms':  round(infer_ms,2),
            })
            log_f.flush()

            if tick % 50 == 0:
                print(f'tick={tick:5d}  pred=({pred_pix[0]:6.1f},{pred_pix[1]:6.1f})'
                      f'  err=({err_x:+.1f},{err_y:+.1f})'
                      f'  Δq=({dq_w:+.4f},{dq_s:+.4f})'
                      f'  {"PAUSED" if paused else "ACTIVE"}')

            # Display
            if preview:
                S = DISPLAY_SCALE
                evt_bgr = cv2.resize(make_event_img(binned),
                                     (CROP_W*S, DAVIS_H*S), interpolation=cv2.INTER_NEAREST)
                aps_bgr = cv2.resize(
                    cv2.cvtColor(last_aps[:,CROP_X0:CROP_X0+CROP_W], cv2.COLOR_GRAY2BGR),
                    (CROP_W*S, DAVIS_H*S), interpolation=cv2.INTER_LINEAR)

                draw_overlays(evt_bgr, pred_pix, cent, err_x, err_y, s=S)
                draw_overlays(aps_bgr, pred_pix, cent, err_x, err_y, s=S)

                state_col = (0,200,0) if not paused else (0,100,255)
                spread_str = f'std=({spread[0]:.0f},{spread[1]:.0f})' if spread else 'std=--'
                vis_str = f'TARGET {spread_str}' if target_visible else f'NO TARGET {spread_str}'
                hud(evt_bgr, [
                    f'tick {tick}   {"PAUSED" if paused else vis_str}',
                    f'pred  ({pred_pix[0]:6.1f}, {pred_pix[1]:6.1f})',
                    f'err   ({err_x:+.1f}, {err_y:+.1f}) px',
                    f'events: {n_ev}   {infer_ms:.0f}ms',
                ], col=state_col if target_visible else (0, 100, 255))
                hud(aps_bgr, [
                    f'gain={args.gain}  maxstep={args.max_step:.3f}rad',
                    f'dq_w={dq_w:+.4f}  dq_s={dq_s:+.4f}',
                    f'waist={q_w}  shoulder={q_s}',
                    f'dcx/w={dcx_per_waist:.0f}  dcy/s={dcy_per_shoulder:.1f}',
                ])

                div  = np.full((DAVIS_H*S, 6, 3), 40, dtype=np.uint8)
                cv2.imshow('Phase 5.3 — Visual Servo', np.hstack([evt_bgr, div, aps_bgr]))

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break
                elif key == ord(' '):
                    paused = not paused
                    print(f'{"PAUSED" if paused else "RESUMED"}')
                elif key in (ord('r'), ord('R')):
                    model.reset_state(1, device=device)
                    print('Model state reset.')
                elif key in (ord('s'), ord('S')):
                    os.makedirs(LOG_DIR, exist_ok=True)
                    fname = os.path.join(LOG_DIR, f'snap_{tick}.png')
                    cv2.imwrite(fname, np.hstack([evt_bgr, div, aps_bgr]))
                    print(f'Saved {fname}')

            tick += 1

    except KeyboardInterrupt:
        print('\nInterrupted.')
    finally:
        _infer_running[0] = False
        cam_running[0]    = False
        log_f.close()
        if preview:
            cv2.destroyAllWindows()
        print('Returning to HOME...')
        arm.goto_home()
        arm.disable()
        print(f'Log → {log_path}')


if __name__ == '__main__':
    main()
