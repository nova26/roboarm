"""
Forward kinematics for WidowX-200 (5 DOF kinematic chain).

Kinematic joint index → DXL servo ID mapping:
  joint 0 → ID1  waist        (z-axis rotation)
  joint 1 → ID2  shoulder     (y-axis rotation; ID3 is mechanical tandem, slaved)
  joint 2 → ID4  elbow        (y-axis rotation)
  joint 3 → ID5  wrist pitch  (y-axis rotation)
  joint 4 → ID6  wrist rotate (x-axis rotation)
  —         ID7  gripper      (not part of kinematic chain)

Angle convention
----------------
  DXL ticks:  0 = 0°, 2048 = 180°, 4095 ≈ 360°
  Kinematic angles (q) are in radians, defined as DEVIATION from HOME ticks.
  q = 0 for all joints → arm is at HOME configuration.

  tick_to_rad(t, joint) = (t - HOME_TICKS[joint]) * TICKS_TO_RAD

Calibration
-----------
  1. HOME_TICKS: tick values at the desired home/rest pose (read from hardware).
  2. HOME_ANGLES_ABS: absolute kinematic angles (rad) at HOME_TICKS.
     Must be calibrated physically. Current defaults are estimates.

Link lengths (from official Trossen WX-200-QS CAD drawing, 7/24/2018)
----------------------------------------------------------------------
  L1 = 0.11325  base to shoulder pivot (vertical)
  SHOULDER_OFFSET = 0.050  horizontal offset from shoulder axis to upper arm pivot
    (the upper arm link is 206.16 mm diagonally: sqrt(50² + 200²))
  L2 = 0.200   shoulder pivot to elbow pivot (vertical component)
  L3 = 0.200   elbow pivot to wrist pitch pivot
  L4 = 0.065   wrist pitch pivot to wrist rotate pivot
  L5 = 0.09358 wrist rotate pivot to TCP (PoE M-matrix TCP: 158.575 mm from wrist pitch)

PoE Kinematic Data (from Trossen spec page)
-------------------------------------------
  M_HOME (EE frame at all-zero joint config):
    [[1,0,0,0.408575],[0,1,0,0],[0,0,1,0.31065],[0,0,0,1]]

  Slist (screw axes in space frame):
    [0, 0, 1,  0,       0,       0    ]  waist
    [0, 1, 0, -0.11065, 0,       0    ]  shoulder  (axis at z=0.11065)
    [0, 1, 0, -0.31065, 0,       0.05 ]  elbow     (axis at x=0.05, z=0.31065)
    [0, 1, 0, -0.31065, 0,       0.25 ]  wrist pitch (axis at x=0.25, z=0.31065)
    [1, 0, 0,  0,       0.31065, 0    ]  wrist rotate
"""

import numpy as np

# ── Physical parameters (official CAD drawing) ────────────────────────────────
L1               = 0.11325   # base to shoulder pivot (m)
SHOULDER_OFFSET  = 0.050     # horizontal offset from shoulder axis to upper arm pivot (m)
L2               = 0.200     # shoulder to elbow, vertical component (m)
L3               = 0.200     # elbow to wrist pitch (m)
L4               = 0.065     # wrist pitch to wrist rotate (m)
L5               = 0.09358   # wrist rotate to TCP (m)

# ── Joint ↔ DXL ID mapping ────────────────────────────────────────────────────
N_JOINTS      = 5
JOINT_TO_DXL  = {0: 1, 1: 2, 2: 4, 3: 5, 4: 6}
DXL_TO_JOINT  = {v: k for k, v in JOINT_TO_DXL.items()}

# ID3 (shoulder2) is tandem with ID2
TANDEM_DXL_ID   = 3
TANDEM_HOME_ID2 = 737
TANDEM_HOME_ID3 = 3353   # 4090 - 737

# ── Tick / angle conversions ──────────────────────────────────────────────────
TICKS_TO_RAD = 2 * np.pi / 4096

# HOME ticks (user-calibrated)
HOME_TICKS = np.array([4090, 736, 948, 1645, 2056], dtype=float)

# Absolute joint angles (rad) at HOME_TICKS in the kinematic frame.
# !! Calibrate these against the physical arm before running OSC. !!
HOME_ANGLES_ABS = np.array([
     0.00,    # joint 0 waist
    -1.40,    # joint 1 shoulder  (estimate: ~−80° from horizontal)
     2.80,    # joint 2 elbow     (estimate: ~+160° cumulative)
     1.57,    # joint 3 wrist     (estimate: ~+90° cumulative)
     0.00,    # joint 4 wrist_rot
], dtype=float)


def ticks_to_rad(ticks):
    """DXL ticks (array length 5) → kinematic angles (rad), relative to HOME."""
    return (np.asarray(ticks, dtype=float) - HOME_TICKS) * TICKS_TO_RAD


def rad_to_ticks(q):
    """Kinematic angles (rad) relative to HOME → DXL ticks (int array)."""
    return np.round(np.asarray(q, dtype=float) / TICKS_TO_RAD + HOME_TICKS).astype(int)


def tandem_id3_ticks(id2_ticks):
    """Return ID3 ticks that mirror ID2 symmetrically about their respective homes."""
    return int(TANDEM_HOME_ID3 + TANDEM_HOME_ID2 - id2_ticks)


# ── Forward kinematics ────────────────────────────────────────────────────────

def fk(q):
    """
    EE position in world frame (x, y, z) in metres.

    q : array-like, shape (5,)
        Kinematic joint angles in radians, RELATIVE to HOME (q=0 at HOME).

    Geometry: the shoulder axis has a 50 mm horizontal structural offset
    (SHOULDER_OFFSET) confirmed by the official CAD drawing.
    """
    q  = np.asarray(q, dtype=float)
    qa = q + HOME_ANGLES_ABS
    q0, q1, q2, q3, _ = qa

    a1 = q1                  # absolute shoulder angle from horizontal
    a2 = q1 + q2             # cumulative to elbow
    a3 = q1 + q2 + q3        # cumulative to wrist

    # Horizontal reach in the arm plane
    # SHOULDER_OFFSET is a fixed structural forward offset at the shoulder
    r = (SHOULDER_OFFSET
       + L2 * np.cos(a1)
       + L3 * np.cos(a2)
       + (L4 + L5) * np.cos(a3))

    # Vertical height
    z = (L1
       + L2 * np.sin(a1)
       + L3 * np.sin(a2)
       + (L4 + L5) * np.sin(a3))

    x = r * np.cos(q0)
    y = r * np.sin(q0)

    return np.array([x, y, z])


def jacobian(q, eps=1e-5):
    """
    Numerical Jacobian J (3×5): ∂EE_pos / ∂q_i.

    q : array-like, shape (5,) in radians relative to HOME
    """
    q   = np.asarray(q, dtype=float)
    ee0 = fk(q)
    J   = np.zeros((3, N_JOINTS))
    for j in range(N_JOINTS):
        dq      = np.zeros(N_JOINTS)
        dq[j]   = eps
        J[:, j] = (fk(q + dq) - ee0) / eps
    return J


# ── Dynamics (approximate) ────────────────────────────────────────────────────

LINK_MASSES  = np.array([0.80, 0.79, 0.32, 0.41, 0.23])
INERTIA_DIAG = np.array([0.10, 0.08, 0.04, 0.04, 0.02])


def inertia_matrix(q):
    """Approximate diagonal joint-space inertia matrix (5×5)."""
    return np.diag(INERTIA_DIAG)


def gravity_bias(q):
    """
    Joint torques (N·m) needed to hold the arm against gravity at configuration q.
    q : array-like, shape (5,) in radians relative to HOME.
    """
    q  = np.asarray(q, dtype=float)
    qa = q + HOME_ANGLES_ABS
    _, q1, q2, q3, _ = qa
    g  = 9.81

    a1 = q1
    a2 = q1 + q2
    a3 = q1 + q2 + q3

    m1, m2, m3, m4, _ = LINK_MASSES

    # Horizontal reach to CoM of each distal link (CoM approximated at link end)
    # Include shoulder offset in reach computation
    c1 = SHOULDER_OFFSET + L2 * np.cos(a1)
    c2 = SHOULDER_OFFSET + L2 * np.cos(a1) + L3 * np.cos(a2)
    c3 = c2 + (L4 + L5) * np.cos(a3)

    tau = np.zeros(N_JOINTS)
    tau[0] = 0.0
    tau[1] = -g * (m2 * c1 + m3 * c2 + m4 * c3)
    tau[2] = -g * (m3 * L3 * np.cos(a2) + m4 * (L3 * np.cos(a2) + (L4 + L5) * np.cos(a3)))
    tau[3] = -g * m4 * (L4 + L5) * np.cos(a3)
    tau[4] = 0.0

    return tau
