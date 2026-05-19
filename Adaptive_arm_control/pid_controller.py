"""
Software PID controller for WidowX-200 joint-space control.

Each joint runs its own PID loop in Python at ~20 Hz.
The PID output is a corrected goal position sent to the Dynamixel servo;
the servo's internal hardware PID then tracks that goal at ~1 kHz.

Control law (per joint, per cycle):
    error      = target_ticks - actual_ticks
    integral  += error * dt
    derivative = (error - prev_error) / dt
    step       = clip(Kp*error + Ki*integral + Kd*derivative, -MAX_STEP, +MAX_STEP)
    goal       = actual_ticks + step

Anti-windup: integral is clamped so Ki*integral never exceeds MAX_STEP.
"""

import time
import numpy as np
import dynamixel_sdk as dxl
from widowx_kinematics import (
    JOINT_TO_DXL, TANDEM_DXL_ID, N_JOINTS,
    rad_to_ticks, ticks_to_rad, HOME_TICKS,
    tandem_id3_ticks,
)

# ── Default gains ──────────────────────────────────────────────────────────────
# Kp: how aggressively to chase the target (ticks of goal movement per tick of error)
# Ki: steady-state correction (start at 0 until Kp/Kd are tuned)
# Kd: damping to reduce overshoot
DEFAULT_GAINS = {
    0: dict(kp=0.8, ki=0.02, kd=0.10),  # waist
    1: dict(kp=0.5, ki=0.04, kd=0.15),  # shoulder (heavy/gravity; low Kp prevents wobble)
    2: dict(kp=0.9, ki=0.05, kd=0.10),  # elbow (needs more drive against gravity)
    3: dict(kp=0.8, ki=0.02, kd=0.08),  # wrist pitch
    4: dict(kp=0.8, ki=0.02, kd=0.06),  # wrist rotate (lighter)
}

# Maximum goal position change per cycle (ticks). Limits max speed.
MAX_STEP = 100          # ~8.8° per cycle at 20 Hz ≈ 176°/s max

# Profile velocity written to servos in PID mode (ticks/sec).
# Must be high enough that the hardware servo can track the software-commanded
# goals at 20 Hz; too low causes lag and wobble.
PID_PROFILE_VEL = 150

ADDR_PROFILE_VEL = 112

# Convergence: all joints within this many ticks for SETTLE_CYCLES in a row
TOLERANCE     = 25      # ~2.2°
SETTLE_CYCLES = 3


class JointPID:
    """Single-joint PID controller operating in tick space."""

    def __init__(self, kp=0.8, ki=0.0, kd=0.1, max_step=MAX_STEP):
        self.kp       = kp
        self.ki       = ki
        self.kd       = kd
        self.max_step = max_step
        self._integral   = 0.0
        self._prev_error = 0.0

    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0

    def compute(self, error, dt):
        """Return goal-position step (ticks) given current error and timestep."""
        self._integral += error * dt
        # Anti-windup: clamp integral contribution
        if self.ki > 0:
            max_i = self.max_step / self.ki
            self._integral = float(np.clip(self._integral, -max_i, max_i))

        derivative       = (error - self._prev_error) / max(dt, 1e-6)
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return float(np.clip(output, -self.max_step, self.max_step))


class ArmPIDController:
    """
    Runs a software PID loop over all 5 kinematic joints.

    Usage
    -----
        ctrl = ArmPIDController(arm)
        ctrl.run_to(q_target_dict, label='POS1', timeout=5.0)
    """

    def __init__(self, arm, gains=None, dt=0.05,
                 tolerance=TOLERANCE, settle_cycles=SETTLE_CYCLES, timeout=6.0,
                 profile_vel=PID_PROFILE_VEL):
        self.arm           = arm
        self.dt            = dt
        self.tolerance     = tolerance
        self.settle_cycles = settle_cycles
        gains = gains or DEFAULT_GAINS
        self.pids = {i: JointPID(**gains[i]) for i in range(N_JOINTS)}

        # Set a fast hardware profile velocity so servos can track 20 Hz goal updates
        ids = list(JOINT_TO_DXL.values()) + [TANDEM_DXL_ID]
        for dxl_id in ids:
            arm.pk.write4ByteTxRx(arm.ph, dxl_id, ADDR_PROFILE_VEL, profile_vel)
        print(f'PID profile_vel={profile_vel} ticks/s set on {len(ids)} servos')

    def _read_ticks(self):
        """Return actual ticks for joints 0-4 as numpy array. Falls back to HOME_TICKS on read failure."""
        ticks = []
        for i in range(N_JOINTS):
            val = self.arm._read_ticks(JOINT_TO_DXL[i])
            ticks.append(float(HOME_TICKS[i]) if (val is None or val != val) else float(val))
        return np.array(ticks)

    def _write_ticks(self, ticks):
        """Send goal positions to all 5 joints + tandem."""
        for i in range(N_JOINTS):
            self.arm._write_ticks(JOINT_TO_DXL[i], int(ticks[i]))
        self.arm._write_ticks(TANDEM_DXL_ID, tandem_id3_ticks(int(ticks[1])))

    def run_to(self, q_dict, label='', timeout=6.0):
        """
        Drive to target joint angles using per-joint PID.

        q_dict   : {joint_idx: angle_rad} relative to HOME
        label    : name for logging
        timeout  : max seconds before giving up
        Returns  : True if converged, False if timeout
        """
        # Build target tick array
        target_ticks = np.array(
            [HOME_TICKS[i] + q_dict.get(i, 0.0) / (2 * np.pi / 4096)
             for i in range(N_JOINTS)],
            dtype=float
        )

        for pid in self.pids.values():
            pid.reset()

        settled   = 0
        t_start   = time.time()
        t_prev    = t_start

        if label:
            print(f'\n→ {label}')

        while True:
            now = time.time()
            dt  = now - t_prev
            t_prev = now

            actual = self._read_ticks()
            errors = target_ticks - actual

            # Check convergence
            if np.all(np.abs(errors) < self.tolerance):
                settled += 1
                if settled >= self.settle_cycles:
                    elapsed = now - t_start
                    print(f'  Converged in {elapsed:.2f}s  '
                          f'max_err={np.max(np.abs(errors)):.0f} ticks')
                    return True
            else:
                settled = 0

            if now - t_start > timeout:
                print(f'  Timeout ({timeout:.1f}s)  '
                      f'errors={np.round(errors).astype(int)}')
                return False

            # Compute PID corrections and send new goals
            steps = np.array([
                self.pids[i].compute(errors[i], max(dt, self.dt))
                for i in range(N_JOINTS)
            ])
            goal_ticks = actual + steps
            self._write_ticks(goal_ticks)

            time.sleep(self.dt)

    def run_sequence(self, sequence, timeout=6.0):
        """
        Run a list of (label, q_dict) targets in order.
        Returns list of (label, converged) results.
        """
        results = []
        for label, q_dict in sequence:
            ok = self.run_to(q_dict, label=label, timeout=timeout)
            results.append((label, ok))
        return results
