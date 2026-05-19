# WidowX-200 (WX200) — Hardware Summary

## Overview

- **Type**: 5 DOF robotic arm + gripper
- **Reach**: 550 mm
- **Total Span**: 1100 mm
- **Rotation**: Full 360° waist rotation
- **Power Supply**: 12V DC (recommended); EXP board screw terminal input

---

## Onboard Controller

**Robotis OpenCM9.04**
- MCU: ARM Cortex-M3 (STM32F103), 32-bit
- Arduino-compatible firmware environment
- Communicates with DYNAMIXEL servos over TTL
- USB interface to host PC

**OpenCM 485 EXP Board** (expansion, mounted alongside)
- Adds RS-485 DYNAMIXEL support
- Additional servo bus ports (JST connectors)
- Screw terminal power input (12V for servos)

---

## Default Servo Configuration

| ID | Motor | Joint | Stall Torque (12V) | Max Speed (12V) |
|---|---|---|---|---|
| 1 | XM430-W350 | Waist | 4.1 Nm | 57 RPM |
| 2 | XM430-W350 | Shoulder | 4.1 Nm | 57 RPM |
| 3 | XM430-W350 | Shadow Shoulder | 4.1 Nm | 57 RPM |
| 4 | XM430-W350 | Elbow | 4.1 Nm | 57 RPM |
| 5 | XM430-W350 | Wrist Angle | 4.1 Nm | 57 RPM |
| 6 | XL430-W250 | Wrist Rotate | 1.0 Nm | 61 RPM |
| 7 | XL430-W250 | Gripper | 1.0 Nm | 61 RPM |

- **Baudrate**: 1 Mbps
- **Resolution**: 4096 positions/revolution
- Torque/speed specs from Robotis datasheets (XM430-W350, XL430-W250)

---

## Joint Limits

| Joint | Min (deg) | Max (deg) | Min (rad) | Max (rad) |
|---|---|---|---|---|
| Waist | -180° | +180° | -3.1416 | +3.1416 |
| Shoulder | -108° | +113° | -1.8850 | +1.9722 |
| Elbow | -108° | +93° | -1.8850 | +1.6231 |
| Wrist Angle | -100° | +123° | -1.7453 | +2.1468 |
| Wrist Rotate | -180° | +180° | -3.1416 | +3.1416 |
| Gripper | 30 mm (closed) | 74 mm (open) | — | — |

---

## Kinematic Properties

**Model**: Product of Exponentials (PoE) — Modern Robotics convention

### Joint Positions (derived from Slist)

| Joint | x (m) | y (m) | z (m) |
|---|---|---|---|
| Waist | 0.0 | 0.0 | 0.0 |
| Shoulder | 0.0 | 0.0 | 0.11065 |
| Elbow | 0.05 | 0.0 | 0.31065 |
| Wrist Angle | 0.25 | 0.0 | 0.31065 |
| Wrist Rotate | — | 0.0 | 0.31065 |
| End-Effector | 0.408575 | 0.0 | 0.31065 |

### M Matrix — End-Effector Home Pose (4×4)

```
M = [1.0   0.0   0.0   0.408575]
    [0.0   1.0   0.0   0.0     ]
    [0.0   0.0   1.0   0.31065 ]
    [0.0   0.0   0.0   1.0     ]
```

### Slist Matrix — Joint Screw Axes (6×5)

Standard form: 6 rows (screw components) × 5 columns (joints).  
Row order: [ωx, ωy, ωz, vx, vy, vz]

```
         J1      J2        J3        J4        J5
ωx  [  0.0     0.0       0.0       0.0       1.0   ]
ωy  [  0.0     1.0       1.0       1.0       0.0   ]
ωz  [  1.0     0.0       0.0       0.0       0.0   ]
vx  [  0.0    -0.11065  -0.31065  -0.31065   0.0   ]
vy  [  0.0     0.0       0.0       0.0       0.31065]
vz  [  0.0     0.0       0.05      0.25      0.0   ]
```

> **Note**: Many libraries (e.g. `modern_robotics`) expect Slist as a (6, n) array.  
> If stored row-per-joint, transpose before use: `Slist = np.array([...]).T`

---

## Drawings & CAD Files

| Resource | File | Link |
|---|---|---|
| Technical Drawing | WidowX-200.pdf | https://docs.trossenrobotics.com/interbotix_xsarms_docs/_downloads/20d7805161fea219a2e6ee9ed519c0d7/WidowX-200.pdf |
| Solid CAD (STEP) | 5_WXA-200-M.zip | https://docs.trossenrobotics.com/interbotix_xsarms_docs/_downloads/65f7535b8419c6ee8ed44742df56800d/5_WXA-200-M.zip |
| Mesh Models (STL) | via GitHub | https://github.com/Interbotix/interbotix_ros_manipulators |

---

## References

- Trossen Docs: https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications/wx200.html
- Robotis OpenCM9.04: https://emanual.robotis.com/docs/en/parts/controller/opencm904/
- Robotis XM430-W350: https://emanual.robotis.com/docs/en/dxl/x/xm430-w350/
- Robotis XL430-W250: https://emanual.robotis.com/docs/en/dxl/x/xl430-w250/
