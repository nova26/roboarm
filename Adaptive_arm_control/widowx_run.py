"""
WidowX-200 movement script.

Modes
-----
    python widowx_run.py                 # HOME → POS1 → waypoints → HOME (joint-space)
    python widowx_run.py --pid           # Same sequence using software PID feedback loop
    python widowx_run.py --osc           # OSC EE-space (x,y,z only; wrist uncontrolled)
    python widowx_run.py --osc --adapt   # OSC + Nengo adaptive compensation
    python widowx_run.py --log           # Any mode + write servo audit log to logs/

Log columns (CSV)
-----------------
    run_ts, waypoint, joint, dxl_id, goal_ticks, actual_ticks, error_ticks,
    hw_error, temperature_C, current_mA
"""

import argparse
import csv
import os
import time
import json
import numpy as np

import dynamixel_sdk as dxl
from widowx_arm import WidowXArm
from widowx_kinematics import TICKS_TO_RAD, HOME_TICKS, JOINT_TO_DXL, N_JOINTS, rad_to_ticks

WAYPOINTS_FILE = os.path.join(os.path.dirname(__file__), '..', 'waypoints.json')
LOGS_DIR       = os.path.join(os.path.dirname(__file__), '..', 'logs')

# ── Dynamixel read addresses used only for logging ────────────────────────────
ADDR_PRESENT_CURR = 126   # 2 bytes, unit 2.69 mA
ADDR_HW_ERROR     = 70    # 1 byte
ADDR_TEMPERATURE  = 146   # 1 byte, °C

HW_ERROR_NAMES = {1: 'Voltage', 4: 'Overheat', 8: 'Encoder',
                  16: 'ElecShock', 32: 'Overload'}

# ── Saved positions (DXL ticks from keyboard_control.py) ─────────────────────
HOME_DXL = {1: 4090, 2: 736,  3: 3353, 4: 948,  5: 1645, 6: 2056, 7: 1448}
POS1_DXL = {1: 4090, 2: 904,  3: 3185, 4: 1564, 5: 1435, 6: 2056, 7: 1448}

_dxl_ids = [1, 2, 4, 5, 6]


def dxl_to_angles(dxl_pos: dict) -> dict:
    """Convert a {dxl_id: ticks} dict to {joint_idx: rad} relative to HOME."""
    return {
        i: (int(dxl_pos[str(_dxl_ids[i])]) - HOME_DXL[_dxl_ids[i]]) * TICKS_TO_RAD
        for i in range(5)
        if str(_dxl_ids[i]) in dxl_pos
    }


POS1_ANGLES = dxl_to_angles({str(k): v for k, v in POS1_DXL.items()})

# ── OSC EE-space target (FK-computed offset HOME → POS1) ─────────────────────
POS1_EE_OFFSET = np.array([-0.1269, 0.0000, -0.1869])


# ── Servo state logger ────────────────────────────────────────────────────────

class ServoLogger:
    """
    Records commanded servo state to a timestamped CSV after each waypoint.

    Note: the USB adapter on this system is write-only (Arduino-based half-duplex
    bridge that does not relay servo status packets back to the host). Actual
    position readback is therefore unavailable; only commanded goal positions
    and timing are logged.
    """

    FIELDS = ['run_ts', 'elapsed_s', 'waypoint', 'joint', 'dxl_id', 'goal_ticks']

    def __init__(self, arm):
        self.arm      = arm
        self.run_ts   = time.strftime('%Y%m%d_%H%M%S')
        self._t0      = time.time()
        os.makedirs(LOGS_DIR, exist_ok=True)
        self.path     = os.path.join(LOGS_DIR, f'run_{self.run_ts}.csv')
        self._f       = open(self.path, 'w', newline='')
        self._w       = csv.DictWriter(self._f, fieldnames=self.FIELDS)
        self._w.writeheader()
        print(f'Logging to {self.path}')

    def record(self, waypoint_label, goal_angles):
        """Write one row per joint with the commanded goal position."""
        goal_ticks_arr = rad_to_ticks(
            np.array([goal_angles.get(i, 0.0) for i in range(N_JOINTS)])
        )
        ts      = time.strftime('%Y-%m-%d %H:%M:%S')
        elapsed = round(time.time() - self._t0, 2)
        for joint_idx in range(N_JOINTS):
            self._w.writerow({
                'run_ts':     ts,
                'elapsed_s':  elapsed,
                'waypoint':   waypoint_label,
                'joint':      joint_idx,
                'dxl_id':     JOINT_TO_DXL[joint_idx],
                'goal_ticks': int(goal_ticks_arr[joint_idx]),
            })
        self._f.flush()

    def close(self):
        self._f.close()
        print(f'Log saved → {self.path}')


# ── Sequence builder ──────────────────────────────────────────────────────────

def build_sequence():
    """Build the full POS1 + waypoints movement sequence."""
    sequence = [('POS1', POS1_ANGLES)]

    if os.path.exists(WAYPOINTS_FILE):
        waypoints = json.load(open(WAYPOINTS_FILE))
        xyz_count = 0
        for i, wp in enumerate(waypoints):
            if not any(k.isdigit() for k in wp):
                print(f'WP{i+1}: skipping — xyz-only format, no tick data')
                continue
            angles = dxl_to_angles(wp)
            label  = f'WP{i+1}'
            if 'x' in wp:
                label += f'  ({wp["x"]:.3f}, {wp["y"]:.3f}, {wp["z"]:.3f}) m'
                xyz_count += 1
            sequence.append((label, angles))
        print(f'Loaded {len(waypoints)} waypoints ({xyz_count} with xyz) from {WAYPOINTS_FILE}')
    else:
        print('No waypoints file — running HOME → POS1 → HOME only')

    return sequence


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid',    action='store_true',
                        help='Use software PID feedback loop')
    parser.add_argument('--osc',    action='store_true',
                        help='Use OSC EE-space controller (wrist orientation uncontrolled)')
    parser.add_argument('--adapt',  action='store_true',
                        help='Enable Nengo adaptive dynamics compensation (requires --osc)')
    parser.add_argument('--log',    action='store_true',
                        help='Write per-waypoint servo audit log to logs/')
    parser.add_argument('--device', default='/dev/ttyACM0')
    parser.add_argument('--steps',  type=int, default=None,
                        help='Max steps per target for OSC mode')
    parser.add_argument('--timeout', type=float, default=10.0,
                        help='Per-waypoint timeout in seconds for PID mode (default: 10.0)')
    args = parser.parse_args()

    HOME_ANGLES = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}

    if args.osc:
        arm = WidowXArm(
            init_angles    = HOME_ANGLES,
            device         = args.device,
            target         = [POS1_EE_OFFSET],
            return_to_null = True,
            th             = 2e-2,
            dt             = 0.05,
            adapt          = args.adapt,
        )
        arm.simulate(steps=args.steps)
        arm.show_monitor()
        return

    arm = WidowXArm(
        init_angles = HOME_ANGLES,
        device      = args.device,
        target      = None,
        adapt       = False,
    )
    sequence  = build_sequence()
    logger    = ServoLogger(arm) if args.log else None

    if args.pid:
        # ── Software PID mode ─────────────────────────────────────────────────
        from pid_controller import ArmPIDController
        ctrl = ArmPIDController(arm, timeout=args.timeout)
        ctrl.run_sequence(sequence + [('HOME', HOME_ANGLES)], timeout=args.timeout)
        arm._disable_torque()

    else:
        # ── Direct joint-space mode (default) ─────────────────────────────────
        try:
            for name, angles in sequence:
                print(f'\n→ Moving to {name}...')
                arm.send_target_angles(angles)
                time.sleep(1.5)
                if logger:
                    logger.record(name, angles)

            print('\n→ Returning to HOME...')
            arm.send_target_angles(HOME_ANGLES)
            time.sleep(3.0)
            if logger:
                logger.record('HOME', HOME_ANGLES)

            print('Done.')
        finally:
            arm._disable_torque()
            if logger:
                logger.close()


if __name__ == '__main__':
    main()
