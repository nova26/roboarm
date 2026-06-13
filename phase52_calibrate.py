"""
Phase 5.2 — Camera-Jacobian calibration.

Measures how many pixels the event centroid shifts per radian of joint motion
for waist (→ x shift) and shoulder (→ y shift).

Procedure:
  1. Arm at HOME, ball in FOV → measure baseline centroid
  2. Rotate waist by ±STEP_DEG, measure centroid shift → dcx_per_waist
  3. Return HOME, rotate shoulder by ±STEP_DEG  → dcy_per_shoulder
  4. Save result to calib.json

Usage:
    python phase52_calibrate.py [--device /dev/ttyACM0] [--step 5]

The ball should be roughly centred in the FOV and stationary during calibration.
"""

import argparse
import datetime
import json
import os
import sys
import time
import threading

import numpy as np
import dv_processing as dv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Adaptive_arm_control'))
import dynamixel_sdk as dxl_sdk

# ── DXL constants (direct writes, no WidowXArm to keep this script standalone) ─
ADDR_TORQUE   = 64
ADDR_PROF_ACC = 108
ADDR_PROF_VEL = 112
ADDR_GOAL_POS = 116
ADDR_PRES_POS = 132

BAUDRATE      = 1_000_000
TICKS_TO_RAD  = 2 * np.pi / 4096

# HOME ticks (calibrated 2026-05-19)
HOME_TICKS = {1: 4090, 2: 736, 4: 948, 5: 1645, 6: 2056}

# Camera constants
DAVIS_W, DAVIS_H = 346, 260
CROP_W    = 260
CROP_X0   = (DAVIS_W - CROP_W) // 2   # 43
BAF_US    = 2000
MEASURE_S = 0.5     # seconds of events to average centroid over
N_TICKS   = 20      # 10ms ticks to average (= 200ms)
SETTLE_S  = 2.5     # wait after each move for arm to settle


# ── camera helpers ─────────────────────────────────────────────────────────────

def open_camera():
    devices = dv.io.camera.discover()
    if not devices:
        raise RuntimeError('No DAVIS camera found.')
    cam = dv.io.camera.open(devices[0].serialNumber)
    print(f'Camera: {cam.getCameraName()}')
    return cam


def measure_centroid(cam, baf, n_ticks=N_TICKS, bin_ms=10.0):
    """Collect events for n_ticks × bin_ms ms and return mean centroid (cx, cy)."""
    centroids = []
    for _ in range(n_ticks):
        t0 = time.monotonic()
        xs, ys = [], []
        while time.monotonic() - t0 < bin_ms / 1000.0:
            b = cam.getNextEventBatch()
            if b and len(b) > 0:
                baf.accept(b)
                f = baf.generateEvents()
                if f and len(f) > 0:
                    coords = np.array(f.coordinates(), dtype=np.int32)
                    mask   = (coords[:, 0] >= CROP_X0) & (coords[:, 0] < CROP_X0 + CROP_W)
                    if mask.any():
                        xs.extend((coords[mask, 0] - CROP_X0).tolist())
                        ys.extend(coords[mask, 1].tolist())
            time.sleep(0.001)
        if xs:
            centroids.append((float(np.mean(xs)), float(np.mean(ys))))

    if not centroids:
        return None
    return (float(np.mean([c[0] for c in centroids])),
            float(np.mean([c[1] for c in centroids])))


# ── arm helpers ────────────────────────────────────────────────────────────────

class ArmDriver:
    def __init__(self, device):
        self.ph = dxl_sdk.PortHandler(device)
        self.pk = dxl_sdk.PacketHandler(2.0)
        self.ph.openPort()
        self.ph.setBaudRate(BAUDRATE)
        self.ph.clearPort()
        time.sleep(0.2)
        self._enable_all()

    def _enable_all(self):
        for ID in HOME_TICKS:
            self.pk.reboot(self.ph, ID); time.sleep(0.12)
        time.sleep(1.5)
        self.ph.clearPort()
        for ID in HOME_TICKS:
            self.pk.write4ByteTxOnly(self.ph, ID, ADDR_PROF_ACC, 5);  time.sleep(0.05)
            self.pk.write4ByteTxOnly(self.ph, ID, ADDR_PROF_VEL, 15); time.sleep(0.05)
            self.pk.write1ByteTxOnly(self.ph, ID, ADDR_TORQUE,   1);  time.sleep(0.05)
        print('Arm: torque enabled')

    def goto_home(self):
        for ID, ticks in HOME_TICKS.items():
            self.pk.write4ByteTxOnly(self.ph, ID, ADDR_GOAL_POS,
                                     int(ticks) & 0xFFFFFFFF)
            time.sleep(0.05)
        time.sleep(SETTLE_S)
        print('Arm: at HOME')

    def move_joint_by_deg(self, joint_id, delta_deg):
        """Move joint_id by delta_deg relative to HOME."""
        delta_ticks = int(round(delta_deg / 360.0 * 4096))
        target = HOME_TICKS[joint_id] + delta_ticks
        self.pk.write4ByteTxOnly(self.ph, joint_id, ADDR_GOAL_POS,
                                 int(target) & 0xFFFFFFFF)
        time.sleep(SETTLE_S)
        return delta_deg * np.pi / 180.0   # return delta in radians

    def disable(self):
        for ID in HOME_TICKS:
            self.pk.write1ByteTxOnly(self.ph, ID, ADDR_TORQUE, 0)
            time.sleep(0.05)
        self.ph.closePort()
        print('Arm: torque disabled')


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='/dev/ttyACM0')
    parser.add_argument('--step',   type=float, default=5.0,
                        help='Joint rotation step in degrees (default 5)')
    args = parser.parse_args()

    cam = open_camera()
    baf = dv.noise.BackgroundActivityNoiseFilter(
        cam.getEventResolution(),
        backgroundActivityDuration=datetime.timedelta(microseconds=BAF_US))

    arm = ArmDriver(args.device)

    results = {}

    try:
        # ── Waist calibration (x axis) ────────────────────────────────────────
        print(f'\n=== Waist calibration  (±{args.step}°) ===')
        print('Keep the ball stationary and roughly centred in the FOV.')
        input('Press Enter when ready...')

        arm.goto_home()
        print('Measuring baseline centroid...')
        c0 = measure_centroid(cam, baf)
        print(f'  Baseline: {c0}')

        print(f'Moving waist +{args.step}°...')
        dq = arm.move_joint_by_deg(1, +args.step)
        c_pos = measure_centroid(cam, baf)
        print(f'  +{args.step}°: {c_pos}')

        arm.goto_home()
        print(f'Moving waist -{args.step}°...')
        arm.move_joint_by_deg(1, -args.step)
        c_neg = measure_centroid(cam, baf)
        print(f'  -{args.step}°: {c_neg}')

        arm.goto_home()

        if c_pos and c_neg:
            dcx_per_waist = (c_pos[0] - c_neg[0]) / (2 * dq)
            print(f'\n  dcx_per_waist = {dcx_per_waist:.2f} px/rad')
            results['dcx_per_waist'] = round(dcx_per_waist, 4)
            results['waist_c0']      = c0
            results['waist_c_pos']   = c_pos
            results['waist_c_neg']   = c_neg
        else:
            print('  WARNING: centroid measurement failed — no events?')

        # ── Shoulder calibration (y axis) ─────────────────────────────────────
        print(f'\n=== Shoulder calibration  (±{args.step}°) ===')
        input('Press Enter when ready...')

        arm.goto_home()
        c0 = measure_centroid(cam, baf)
        print(f'  Baseline: {c0}')

        print(f'Moving shoulder +{args.step}°...')
        dq = arm.move_joint_by_deg(2, +args.step)
        c_pos = measure_centroid(cam, baf)
        print(f'  +{args.step}°: {c_pos}')

        arm.goto_home()
        print(f'Moving shoulder -{args.step}°...')
        arm.move_joint_by_deg(2, -args.step)
        c_neg = measure_centroid(cam, baf)
        print(f'  -{args.step}°: {c_neg}')

        arm.goto_home()

        if c_pos and c_neg:
            dcy_per_shoulder = (c_pos[1] - c_neg[1]) / (2 * dq)
            print(f'\n  dcy_per_shoulder = {dcy_per_shoulder:.2f} px/rad')
            results['dcy_per_shoulder'] = round(dcy_per_shoulder, 4)
            results['shoulder_c0']      = c0
            results['shoulder_c_pos']   = c_pos
            results['shoulder_c_neg']   = c_neg
        else:
            print('  WARNING: centroid measurement failed — no events?')

    finally:
        arm.disable()

    # ── Save calibration ──────────────────────────────────────────────────────
    results['step_deg']    = args.step
    results['image_hw']    = [DAVIS_H, CROP_W]
    results['image_centre'] = [CROP_W / 2, DAVIS_H / 2]
    results['timestamp']   = time.strftime('%Y-%m-%d %H:%M:%S')

    out_path = os.path.join(os.path.dirname(__file__), 'calib.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\n{"="*50}')
    print(f'Calibration saved → {out_path}')
    if 'dcx_per_waist' in results:
        print(f'  dcx_per_waist    = {results["dcx_per_waist"]:+.2f} px/rad')
    if 'dcy_per_shoulder' in results:
        print(f'  dcy_per_shoulder = {results["dcy_per_shoulder"]:+.2f} px/rad')
    print(f'  image_centre     = {results["image_centre"]}')
    print('='*50)


if __name__ == '__main__':
    main()
