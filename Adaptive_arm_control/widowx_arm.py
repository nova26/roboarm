"""
WidowXArm — real-hardware replacement for the MuJoCo Simulation class.

Provides the identical API used by OSC and DynamicsAdaptation so that the
control pipeline (OSC → adaptive_control) works without modification.

Control approach
----------------
  OSC generates joint-space torques u [N·m].
  These are converted to position increments via simplified dynamics:
      Δq = M⁻¹ · u · dt
  and sent as DYNAMIXEL goal positions.

  This approximates impedance control on a position-controlled arm.
  Accuracy improves when INERTIA_DIAG in widowx_kinematics is well-tuned.

Usage
-----
  from widowx_arm import WidowXArm
  import numpy as np

  # Angles relative to HOME (q=0 = stay at home)
  init_angles = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}

  targets = [np.array([0.10, 0.05, 0.10]),   # offsets from home EE position
             np.array([0.10, 0.05, -0.10])]

  arm = WidowXArm(init_angles, target=targets, adapt=False)
  arm.simulate()
  arm.show_monitor()
"""

import time
import numpy as np
import dynamixel_sdk as dxl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from widowx_kinematics import (
    fk, jacobian, inertia_matrix, gravity_bias,
    ticks_to_rad, rad_to_ticks, tandem_id3_ticks,
    JOINT_TO_DXL, N_JOINTS,
    TANDEM_DXL_ID,
)
from OSC import OSC

# ── DYNAMIXEL control table addresses ─────────────────────────────────────────
ADDR_TORQUE_ENABLE = 64
ADDR_HW_ERROR      = 70
ADDR_OPERATING_MODE = 11
ADDR_PROFILE_ACC   = 108
ADDR_PROFILE_VEL   = 112
ADDR_GOAL_POS      = 116
ADDR_PRESENT_POS   = 132

# Dynamixel operating modes
OP_POSITION_CONTROL  = 3   # 0–4095 ticks only
OP_EXTENDED_POSITION = 4   # multi-turn, supports negative ticks

# Joints that need extended position mode (those with negative goal positions)
EXTENDED_POSITION_IDS = {JOINT_TO_DXL[0]}  # waist (ID1) goes negative

# Profile velocity sent to each servo at startup (ticks/sec; lower = safer)
STARTUP_PROFILE_VEL = 40

# Maximum position change per control step (rad); prevents violent movements
MAX_STEP_RAD = 0.04   # ~2.3°


class WidowXArm:
    """
    Real WidowX-200 hardware interface. Drop-in for the MuJoCo Simulation class.

    Parameters
    ----------
    init_angles : dict {joint_idx: angle_rad}
        Home configuration in radians relative to HOME_TICKS. Use all-zeros
        to start at the hardware-calibrated HOME position.
    device : str
        Serial port for the USB2DXL bridge.
    baudrate : int
        DXL bus baud rate (must match servo EEPROM setting).
    target : list of np.ndarray, optional
        3-D target offsets (m) from the home EE position.
    return_to_null : bool
        If True, return to home between consecutive targets.
    th : float
        Success threshold distance (m) to declare target reached.
    dt : float
        Control loop timestep (s). Smaller = more responsive but heavier CPU.
    adapt : bool
        Enable Nengo adaptive dynamics compensation.
    """

    def __init__(self,
                 init_angles,
                 device    = '/dev/ttyACM0',
                 baudrate  = 1_000_000,
                 target    = None,
                 return_to_null = False,
                 th        = 2e-2,
                 dt        = 0.05,
                 adapt     = False):

        self.init_angles   = init_angles
        self.target        = target
        self.return_to_null = return_to_null
        self.th            = th
        self.dt            = dt
        self.adaptation    = adapt
        self.n_joints      = N_JOINTS
        self.model         = type('_M', (), {'n_joints': N_JOINTS})()  # OSC compat shim

        # Open DXL port --------------------------------------------------------
        self.ph = dxl.PortHandler(device)
        self.pk = dxl.PacketHandler(2.0)
        if not self.ph.openPort():
            raise RuntimeError(f'Failed to open {device}')
        self.ph.setBaudRate(baudrate)
        print(f'Connected to {device} @ {baudrate} bps')

        # Clear HW errors, set profile velocity, enable torque -----------------
        self._startup()

        # Position cache — updated on every write; avoids TxRx reads that
        # corrupt the DXL bus state on this write-only OpenCM9.04 bridge.
        from widowx_kinematics import HOME_TICKS
        self._q_cache = ticks_to_rad(np.array(list(HOME_TICKS), dtype=float))

        # Move to home, record null EE position --------------------------------
        self.goto_null_position()
        self.null_position = self.get_ee_position()
        print(f'Home EE position (null): {np.round(self.null_position, 4)} m')

        # Velocity estimation state --------------------------------------------
        self._prev_q  = self._q_cache.copy()
        self._prev_t  = time.time()

        # Controllers ----------------------------------------------------------
        if adapt or target is not None:
            self.controller = OSC(self)

        if adapt:
            from adaptive_control import DynamicsAdaptation
            self.adapt_controller = DynamicsAdaptation(
                n_input           = 10,
                n_output          = 5,
                n_neurons         = 5000,
                n_ensembles       = 5,
                pes_learning_rate = 1e-4,
                means     = [0.12,  2.14,  1.87,  4.32, 0.59,
                             0.12, -0.38, -0.42, -0.29, 0.36],
                variances = [0.08,  0.60,  0.70,  0.30, 0.60,
                             0.08,  1.40,  1.60,  0.70, 1.20],
            )

        # Monitor data ---------------------------------------------------------
        if target is not None:
            self.monitor_dict = {}
            for i, t in enumerate(target):
                self.monitor_dict[i] = {
                    'error':       [],
                    'ee':          [],
                    'q':           [],
                    'dq':          [],
                    'steps':       0,
                    'target':      t,
                    'target_real': None,
                }

    # ── Hardware helpers ───────────────────────────────────────────────────────

    def _startup(self):
        """Reboot servos to clear errors, then set profile and enable torque."""
        ids = list(JOINT_TO_DXL.values()) + [TANDEM_DXL_ID]

        # Reboot clears overload/error state from previous runs
        for dxl_id in ids:
            self.pk.reboot(self.ph, dxl_id)
            time.sleep(0.1)
        time.sleep(1.5)
        self.ph.clearPort()

        for dxl_id in ids:
            self.pk.write4ByteTxOnly(self.ph, dxl_id, ADDR_PROFILE_ACC, 5)
            time.sleep(0.05)
            self.pk.write4ByteTxOnly(self.ph, dxl_id, ADDR_PROFILE_VEL, STARTUP_PROFILE_VEL)
            time.sleep(0.05)
            self.pk.write1ByteTxOnly(self.ph, dxl_id, ADDR_TORQUE_ENABLE, 1)
            time.sleep(0.05)
        print(f'Torque enabled on {len(ids)} servos')

    def _write_ticks(self, dxl_id, ticks):
        self.pk.write4ByteTxOnly(self.ph, dxl_id, ADDR_GOAL_POS, int(ticks) & 0xFFFFFFFF)
        time.sleep(0.03)  # let OpenCM9.04 forward each packet before the next

    # ── Simulation API (same as MuJoCo Simulation class) ──────────────────────

    def get_angles_array(self):
        """Commanded joint angles from cache (no hardware read — bus is write-only)."""
        return self._q_cache.copy()

    def get_angles(self):
        """Joint angles as dict {joint_idx: rad} — matches Simulation.get_angles()."""
        q = self.get_angles_array()
        return {i: float(q[i]) for i in range(N_JOINTS)}

    def get_velocity(self):
        """Joint velocities as dict {joint_idx: rad/s} via finite difference."""
        q   = self.get_angles_array()
        now = time.time()
        dt  = max(now - self._prev_t, 1e-3)
        dq  = (q - self._prev_q) / dt
        self._prev_q = q
        self._prev_t = now
        return {i: float(dq[i]) for i in range(N_JOINTS)}

    def get_ee_position(self):
        """EE position [x, y, z] in metres from forward kinematics."""
        return fk(self.get_angles_array())

    def get_Jacobian(self):
        """6×N Jacobian (position rows; orientation rows zero-filled)."""
        J6        = np.zeros((6, N_JOINTS))
        J6[:3, :] = jacobian(self.get_angles_array())
        return J6

    def get_inertia_matrix(self):
        """Joint-space inertia matrix (N×N)."""
        return inertia_matrix(self.get_angles_array())

    def get_gravity_bias(self):
        """Gravity torques (N,) in joint space."""
        return gravity_bias(self.get_angles_array())

    def send_target_angles(self, q_dict):
        """
        Send absolute goal positions to hardware.
        q_dict : {joint_idx: angle_rad} relative to HOME.
        """
        q_target = self._q_cache.copy()
        for idx, val in q_dict.items():
            q_target[idx] = val

        ticks = rad_to_ticks(q_target)
        for joint_idx in range(N_JOINTS):
            self._write_ticks(JOINT_TO_DXL[joint_idx], ticks[joint_idx])

        # Keep ID3 tandem with ID2
        self._write_ticks(TANDEM_DXL_ID, tandem_id3_ticks(int(ticks[1])))
        self._q_cache = q_target

    def send_forces(self, u):
        """
        Convert joint-space torques u [N·m] to position commands.

        Δq = M⁻¹ · u · dt   (simplified torque integration)
        goal = current_q + Δq, clamped to MAX_STEP_RAD per step.
        """
        q     = self._q_cache.copy()
        M_inv = np.linalg.inv(inertia_matrix(q))
        dq    = np.clip(M_inv @ u[:N_JOINTS] * self.dt, -MAX_STEP_RAD, MAX_STEP_RAD)
        q_new = q + dq

        ticks = rad_to_ticks(q_new)
        for joint_idx in range(N_JOINTS):
            self._write_ticks(JOINT_TO_DXL[joint_idx], ticks[joint_idx])

        self._write_ticks(TANDEM_DXL_ID, tandem_id3_ticks(int(ticks[1])))
        self._q_cache = q_new

    def goto_null_position(self):
        """Move to the home configuration defined by init_angles."""
        self.send_target_angles(self.init_angles)
        time.sleep(1.5)

    def run_sequence_abs(self, targets_abs, names=None, steps=300):
        """
        Drive to each absolute EE target [x, y, z] (metres) using OSC.
        Disables torque and closes port when done.
        """
        if not hasattr(self, 'controller'):
            self.controller = OSC(self)

        try:
            for i, target in enumerate(targets_abs):
                label  = names[i] if names else f'Target {i + 1}'
                target = np.asarray(target, dtype=float)
                print(f'\n→ {label}: {np.round(target, 4)} m')

                error = float('inf')
                step  = 0
                while error > self.th:
                    step += 1
                    if steps is not None and step > steps:
                        print(f'  Timeout ({step} steps), error={error:.4f} m')
                        break

                    position = self.get_angles()
                    velocity = self.get_velocity()
                    u        = self.controller.generate(position, velocity, target)
                    self.send_forces(u)

                    ee    = self.get_ee_position()
                    error = float(np.linalg.norm(target - ee))
                    time.sleep(self.dt)
                else:
                    print(f'  Reached in {step} steps (error={error:.4f} m)')
        finally:
            self._disable_torque()

    # ── Main control loop ─────────────────────────────────────────────────────

    def simulate(self, steps=None):
        """
        Run the OSC (+ optional adaptive) controller to reach each target.

        steps : int, optional
            Maximum iterations per target. None = run until threshold reached.
        """
        if self.target is None:
            print('No target defined.')
            return

        try:
            for exp in self.monitor_dict:
                target = self.null_position + self.monitor_dict[exp]['target']
                self.monitor_dict[exp]['target_real'] = np.copy(target[:3])
                print(f'\n→ Target {exp}: {np.round(target[:3], 4)} m')

                step  = 0
                error = float('inf')

                while True:
                    step += 1
                    if steps is not None and step > steps:
                        self.monitor_dict[exp]['steps'] = step
                        break

                    if error < self.th:
                        print(f'  Reached in {step} steps (error={error:.4f} m)')
                        self.monitor_dict[exp]['steps'] = step
                        if self.return_to_null:
                            self.goto_null_position()
                        break

                    position = self.get_angles()
                    velocity = self.get_velocity()

                    u = self.controller.generate(position, velocity, target)

                    if self.adaptation:
                        pos_arr = np.array([position[i] for i in range(5)])
                        vel_arr = np.array([velocity[i] for i in range(5)])
                        u_adapt           = np.zeros(N_JOINTS)
                        u_adapt[:5]       = self.adapt_controller.generate(
                            input_signal    = np.hstack((pos_arr, vel_arr)),
                            training_signal = np.array(self.controller.training_signal[:5]),
                        )
                        u += u_adapt

                    self.send_forces(u)

                    ee  = self.get_ee_position()
                    error = float(np.sqrt(np.sum((target[:3] - ee) ** 2)))

                    self.monitor_dict[exp]['error'].append(error)
                    self.monitor_dict[exp]['ee'].append(np.copy(ee))
                    self.monitor_dict[exp]['q'].append([position[i] for i in range(N_JOINTS)])
                    self.monitor_dict[exp]['dq'].append([velocity[i] for i in range(N_JOINTS)])

                    time.sleep(self.dt)

        finally:
            self._disable_torque()

    def _disable_torque(self):
        for dxl_id in list(JOINT_TO_DXL.values()) + [TANDEM_DXL_ID]:
            self.pk.write1ByteTxOnly(self.ph, dxl_id, ADDR_TORQUE_ENABLE, 0)
        time.sleep(0.1)
        self.ph.closePort()
        print('Torque disabled, port closed.')

    # ── Monitoring ────────────────────────────────────────────────────────────

    def show_monitor(self):
        """Plot convergence and EE trajectory for each target (mirrors original)."""
        if not hasattr(self, 'monitor_dict'):
            print('No monitor data.')
            return

        for exp in self.monitor_dict:
            d = self.monitor_dict[exp]
            if not d['error']:
                continue

            dist_covered = np.sqrt(np.sum(
                (d['target_real'] - d['ee'][0]) ** 2)) if d['ee'] else 0

            print(f'Target {exp}: distance={dist_covered:.4f} m, '
                  f'final error={d["error"][-1]:.4f} m, steps={d["steps"]}')

            plt.figure()
            plt.plot(d['error'])
            plt.title(f'Target {exp} — distance to target')
            plt.xlabel('Step')
            plt.ylabel('Error (m)')
            plt.tight_layout()
            plt.show()

            if d['ee']:
                ax = plt.figure().add_subplot(111, projection='3d')
                ee = np.array(d['ee'])
                ax.plot(ee[:, 0], ee[:, 1], ee[:, 2], label='EE path')
                ax.scatter(*d['target_real'], c='r', s=80, label='Target')
                ax.set_title(f'Target {exp} — EE trajectory')
                ax.legend()
                plt.tight_layout()
                plt.show()
