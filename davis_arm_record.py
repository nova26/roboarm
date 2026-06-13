"""
Record DAVIS346 events + frames while the arm executes its waypoint sequence.

Output (saved to recordings/<timestamp>/):
    events.npz      — structured array: x, y, t (µs), p (0/1)
    frames.npz      — array (N, H, W) uint8 + frame timestamps
    waypoints.json  — {waypoint_label: camera_timestamp_us} for segmentation

Usage:
    python davis_arm_record.py
    python davis_arm_record.py --device /dev/ttyACM0
    python davis_arm_record.py --preview     # show live feed while recording
"""

import argparse
import json
import os
import sys
import threading
import time

import numpy as np
import dv_processing as dv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Adaptive_arm_control'))
from widowx_arm import WidowXArm
from widowx_run import build_sequence, HOME_DXL, POS1_DXL

# ── Recording state shared between threads ─────────────────────────────────────

class Recorder:
    def __init__(self):
        self.events_x  = []
        self.events_y  = []
        self.events_t  = []
        self.events_p  = []
        self.frames    = []
        self.frame_ts  = []
        self.waypoints = {}   # label → camera timestamp µs
        self._lock     = threading.Lock()
        self._running  = False

    def mark_waypoint(self, label):
        """Call from arm thread to tag the current camera timestamp."""
        with self._lock:
            if self.events_t:
                ts = int(self.events_t[-1][-1])   # last ts of last batch
            else:
                ts = int(time.time() * 1e6)
            self.waypoints[label] = ts
        print(f'  [cam] marked "{label}" @ t={ts}')

    def record_loop(self, cam, preview=False):
        """Camera thread: read events + frames until _running is False."""
        res = cam.getEventResolution()
        W, H = res

        if preview:
            import cv2
            acc = dv.Accumulator(res)
            cv2.namedWindow('Recording', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Recording', W * 2 + 6, H)
            last_gray = np.full((H, W), 128, dtype=np.uint8)
            t_disp = time.monotonic()

        self._running = True
        while self._running:
            batch = cam.getNextEventBatch()
            if batch is not None and len(batch) > 0:
                coords = np.array(batch.coordinates(), dtype=np.int16)
                pols   = np.array(batch.polarities(),  dtype=np.bool_)
                ts_arr = np.array(batch.timestamps(),  dtype=np.int64)
                with self._lock:
                    self.events_x.append(coords[:, 0])
                    self.events_y.append(coords[:, 1])
                    self.events_t.append(ts_arr)
                    self.events_p.append(pols)
                if preview:
                    acc.accept(batch)

            frame = cam.getNextFrame()
            if frame is not None:
                img = frame.image
                gray = img if img.ndim == 2 else img[:, :, 0]
                with self._lock:
                    self.frames.append(gray.copy())
                    self.frame_ts.append(frame.timestamp)
                if preview:
                    last_gray = gray

            if preview:
                now = time.monotonic()
                if (now - t_disp) * 1000 >= 33:
                    t_disp = now
                    ef = acc.generateFrame()
                    evt_bgr  = cv2.cvtColor(ef.image, cv2.COLOR_GRAY2BGR)
                    gray_bgr = cv2.cvtColor(last_gray, cv2.COLOR_GRAY2BGR)
                    div      = np.full((H, 6, 3), 80, dtype=np.uint8)
                    cv2.putText(evt_bgr,  'Events', (8, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 1)
                    cv2.putText(gray_bgr, 'Frame',  (8, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 1)
                    cv2.imshow('Recording', np.hstack([evt_bgr, div, gray_bgr]))
                    cv2.waitKey(1)
                    acc.clear()

        if preview:
            cv2.destroyAllWindows()

    def stop(self):
        self._running = False

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)

        # Events
        if self.events_x:
            x = np.concatenate(self.events_x).astype(np.int16)
            y = np.concatenate(self.events_y).astype(np.int16)
            t = np.concatenate(self.events_t).astype(np.int64)
            p = np.concatenate(self.events_p).astype(np.bool_)
            np.savez_compressed(os.path.join(out_dir, 'events.npz'),
                                x=x, y=y, t=t, p=p)
            print(f'  Events:  {len(x):,} saved')
        else:
            print('  Events:  none')

        # Frames
        if self.frames:
            frames_arr = np.stack(self.frames, axis=0)
            ts_arr     = np.array(self.frame_ts, dtype=np.int64)
            np.savez_compressed(os.path.join(out_dir, 'frames.npz'),
                                frames=frames_arr, timestamps=ts_arr)
            print(f'  Frames:  {len(self.frames)} saved  shape={frames_arr.shape}')
        else:
            print('  Frames:  none')

        # Waypoint timestamps
        wp_path = os.path.join(out_dir, 'waypoints.json')
        with open(wp_path, 'w') as f:
            json.dump(self.waypoints, f, indent=2)
        print(f'  Waypoints: {len(self.waypoints)} markers saved')
        print(f'  Output: {out_dir}')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',  default='/dev/ttyACM0')
    parser.add_argument('--preview', action='store_true',
                        help='Show live camera feed while recording')
    parser.add_argument('--dwell',   type=float, default=1.5,
                        help='Seconds to dwell at each waypoint (default 1.5)')
    args = parser.parse_args()

    # ── Open camera ───────────────────────────────────────────────────────────
    devices = dv.io.camera.discover()
    if not devices:
        print('ERROR: No DAVIS camera found.'); return
    cam = dv.io.camera.open(devices[0].serialNumber)
    print(f'Camera: {cam.getCameraName()}  {cam.getEventResolution()}')

    rec = Recorder()
    cam_thread = threading.Thread(
        target=rec.record_loop, args=(cam, args.preview), daemon=True)

    # ── Open arm ──────────────────────────────────────────────────────────────
    HOME_ANGLES = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
    arm = WidowXArm(init_angles=HOME_ANGLES, device=args.device, target=None)
    sequence = build_sequence()

    out_dir = os.path.join(os.path.dirname(__file__), 'recordings',
                           time.strftime('%Y%m%d_%H%M%S'))

    # ── Record ────────────────────────────────────────────────────────────────
    print('\nStarting camera recording...')
    cam_thread.start()
    time.sleep(0.3)   # let camera buffer fill

    try:
        for name, angles in sequence:
            print(f'\n→ Moving to {name}...')
            rec.mark_waypoint(name)
            arm.send_target_angles(angles)
            time.sleep(args.dwell)

        print('\n→ Returning to HOME...')
        rec.mark_waypoint('HOME')
        arm.send_target_angles(HOME_ANGLES)
        time.sleep(3.0)

    finally:
        arm._disable_torque()

    # ── Stop and save ─────────────────────────────────────────────────────────
    print('\nStopping camera...')
    rec.stop()
    cam_thread.join(timeout=2.0)

    print('Saving recording...')
    rec.save(out_dir)
    print('Done.')


if __name__ == '__main__':
    main()
