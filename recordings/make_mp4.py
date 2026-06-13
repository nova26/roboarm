"""
Convert a davis_arm_record recording to MP4.

Produces a side-by-side video: event accumulation | APS frame.
Events are accumulated between consecutive APS frame timestamps.

Usage:
    python make_mp4.py <recording_dir>
    python make_mp4.py recordings/20260613_174324
"""

import sys, os, json
import numpy as np
import cv2

def main():
    rec_dir = sys.argv[1] if len(sys.argv) > 1 else '.'

    # ── Load data ──────────────────────────────────────────────────────────────
    ev       = np.load(os.path.join(rec_dir, 'events.npz'))
    ex, ey   = ev['x'].astype(np.int32), ev['y'].astype(np.int32)
    et, ep   = ev['t'].astype(np.int64),  ev['p'].astype(bool)

    fr       = np.load(os.path.join(rec_dir, 'frames.npz'))
    frames   = fr['frames']          # (N, H, W) uint8
    frame_ts = fr['timestamps'].astype(np.int64)

    with open(os.path.join(rec_dir, 'waypoints.json')) as f:
        waypoints = json.load(f)
    # Build reverse lookup: ts → label (for overlay)
    wp_by_ts = {v: k for k, v in waypoints.items()}

    N, H, W = frames.shape
    fps      = len(frames) / ((frame_ts[-1] - frame_ts[0]) / 1e6)
    out_path = os.path.join(rec_dir, 'recording.mp4')

    print(f'Frames : {N}  ({W}x{H})  fps≈{fps:.1f}')
    print(f'Events : {len(et):,}')
    print(f'Output : {out_path}')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W * 2 + 6, H))

    # Find the current waypoint label for each frame
    wp_labels  = sorted(waypoints.items(), key=lambda x: x[1])
    ev_idx     = 0   # pointer into event arrays

    for i, (gray, ts) in enumerate(zip(frames, frame_ts)):
        # ── Accumulate events between previous and current frame ──────────────
        t0 = frame_ts[i - 1] if i > 0 else et[0]
        t1 = ts

        mask = (et >= t0) & (et < t1)
        evt_img = np.full((H, W, 3), 64, dtype=np.uint8)   # gray background
        if mask.any():
            xs, ys, ps = ex[mask], ey[mask], ep[mask]
            evt_img[ys[ps],  xs[ps]]  = (0, 220, 0)    # ON  → green
            evt_img[ys[~ps], xs[~ps]] = (0, 0, 220)    # OFF → red

        # ── APS frame ─────────────────────────────────────────────────────────
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # ── Waypoint label overlay ────────────────────────────────────────────
        current_wp = ''
        for label, wp_ts in wp_labels:
            if ts >= wp_ts:
                current_wp = label
        if current_wp:
            cv2.putText(evt_img,  current_wp, (8, H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
            cv2.putText(gray_bgr, current_wp, (8, H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

        cv2.putText(evt_img,  'Events', (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(gray_bgr, 'Frame',  (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        divider  = np.full((H, 6, 3), 40, dtype=np.uint8)
        combined = np.hstack([evt_img, divider, gray_bgr])
        writer.write(combined)

        if i % 50 == 0:
            print(f'  {i}/{N} frames...', end='\r')

    writer.release()
    print(f'\nDone → {out_path}')


if __name__ == '__main__':
    main()
