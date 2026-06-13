"""
DAVIS346 live viewer — events + grayscale frames side by side.

Usage:
    python davis_viewer.py

Keys:
    Q / ESC  quit
    S        save current combined image to davis_<timestamp>.png
"""

import time
import numpy as np
import cv2
import dv_processing as dv

ACCUMULATION_MS = 33   # event window per display frame (~30 fps)
WINDOW = 'DAVIS346 — Events | Frame'


def main():
    devices = dv.io.camera.discover()
    if not devices:
        print('No DAVIS camera found.')
        return

    cam = dv.io.camera.open(devices[0].serialNumber)
    res = cam.getEventResolution()   # (width, height)
    W, H = res
    print(f'Opened {cam.getCameraName()}  {W}x{H}')
    print('Press Q or ESC to quit, S to save.')

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, W * 2 + 6, H)

    acc        = dv.Accumulator(res)
    last_frame = np.full((H, W), 128, dtype=np.uint8)
    t_acc      = time.monotonic()

    while True:
        # Accumulate events
        batch = cam.getNextEventBatch()
        if batch is not None and len(batch) > 0:
            acc.accept(batch)

        # Grab latest grayscale frame
        frame = cam.getNextFrame()
        if frame is not None:
            img = frame.image
            last_frame = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Render at ~30 fps
        now = time.monotonic()
        if (now - t_acc) * 1000 >= ACCUMULATION_MS:
            t_acc = now

            # Event accumulation → BGR (white=ON, black=OFF, gray=no events)
            acc_frame = acc.generateFrame()
            evt_gray  = acc_frame.image                          # (H, W) uint8
            evt_bgr   = cv2.cvtColor(evt_gray, cv2.COLOR_GRAY2BGR)
            acc.clear()

            # Grayscale frame → BGR
            gray_bgr = cv2.cvtColor(last_frame, cv2.COLOR_GRAY2BGR)

            # Labels
            cv2.putText(evt_bgr,  'Events', (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 1)
            cv2.putText(gray_bgr, 'Frame',  (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 1)

            divider  = np.full((H, 6, 3), 80, dtype=np.uint8)
            combined = np.hstack([evt_bgr, divider, gray_bgr])
            cv2.imshow(WINDOW, combined)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (ord('s'), ord('S')):
            fname = f'davis_{int(time.time())}.png'
            cv2.imwrite(fname, combined)
            print(f'Saved {fname}')

    cv2.destroyAllWindows()
    print('Done.')


if __name__ == '__main__':
    main()
