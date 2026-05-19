import sys, os, tty, termios, select, time, json
import dynamixel_sdk as dxl
from collections import deque

try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Adaptive_arm_control'))
    import numpy as _np
    from widowx_kinematics import fk as _fk, ticks_to_rad as _ticks_to_rad, HOME_TICKS as _HOME_TICKS
    _FK_AVAILABLE = True
except ImportError:
    _FK_AVAILABLE = False


def _compute_ee(positions):
    """Return [x, y, z] metres for current positions dict, or None if FK unavailable."""
    if not _FK_AVAILABLE:
        return None
    _joint_to_dxl = {0: 1, 1: 2, 2: 4, 3: 5, 4: 6}
    ticks = _np.array(
        [positions.get(_joint_to_dxl[i], _HOME_TICKS[i]) for i in range(5)],
        dtype=float
    )
    return _fk(_ticks_to_rad(ticks))

WAYPOINTS_FILE = os.path.join(os.path.dirname(__file__), 'waypoints.json')

DEVICE   = '/dev/ttyACM0'
BAUDRATE = 1_000_000

HOME = {1: 4090, 2: 736, 3: 3353, 4: 948, 5: 1645, 6: 2056, 7: 1448}
POS1 = {1: 4090, 2: 904, 3: 3185, 4: 1564, 5: 1435, 6: 2056, 7: 1448}

SAVED_POSES = {1: ('POS1', POS1)}

LIMITS = {
    1: (2363, 5818),   # obstacle limits (physically measured 2026-05-14)
    2: (  70, 1500),   # Shoulder master
    3: (2590, 4020),   # Shoulder2 slave (tandem)
    4: (   0, 2745),
    5: (   0, 3100),
    6: ( 100, 2800),
    7: (1490, 2600),
}
VELOCITY    = {1: 20, 2: 15, 3: 15, 4: 20, 5: 20, 6: 20, 7: 15}
PROFILE_ACC = 5
TANDEM_SUM  = 4089

ALL_JOINTS  = [1, 2, 3, 4, 5, 6, 7]
CTRL_JOINTS = [1, 2, 4, 5, 6, 7]   # ID3 is tandem slave

JOINT_NAMES = {
    1: 'Waist      ',
    2: 'Shoulder   ',
    3: 'Shoulder2* ',
    4: 'Elbow      ',
    5: 'Wrist Pitch',
    6: 'Wrist Rot. ',
    7: 'Gripper    ',
}

HW_ERRORS = {1:'Voltage', 4:'Overheat', 8:'Encoder', 16:'ElecShock', 32:'Overload'}

ADDR_TORQUE        = 64
ADDR_PROF_ACC      = 108
ADDR_VELOCITY      = 112
ADDR_GOAL_POS      = 116
ADDR_PRES_POS      = 132
ADDR_HW_ERROR      = 70
ADDR_OPER_MODE     = 11

OP_POSITION_CONTROL  = 3   # 0–4095 ticks
OP_EXTENDED_POSITION = 4   # multi-turn, supports negative ticks
# ID1 (waist) has negative goal positions — needs extended position mode
EXTENDED_IDS = {1}

portHandler   = dxl.PortHandler(DEVICE)
packetHandler = dxl.PacketHandler(2.0)
log = deque(maxlen=6)

# ── ANSI helpers ───────────────────────────────────────────────────────────────
CLEAR    = '\033[2J\033[H'
NL       = '\r\n'
BOLD     = '\033[1m'
DIM      = '\033[2m'
REVERSE  = '\033[7m'
RESET    = '\033[0m'
GREEN    = '\033[32m'
RED      = '\033[31m'
CYAN     = '\033[36m'
YELLOW   = '\033[33m'

def goto(r, c): return f'\033[{r};{c}H'
def clr_line():  return '\033[K'


def logmsg(msg):
    log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ── DXL helpers ────────────────────────────────────────────────────────────────
def get_pos(ID):
    pos, r, _ = packetHandler.read4ByteTxRx(portHandler, ID, ADDR_PRES_POS)
    if r != dxl.COMM_SUCCESS:
        return None
    if pos > 0x7FFF_FFFF:
        pos -= 0x1_0000_0000
    return pos


def tandem_id3(id2_pos):
    lo, hi = LIMITS[3]
    return max(lo, min(hi, TANDEM_SUM - id2_pos))


def set_pos(ID, pos):
    lo, hi = LIMITS[ID]
    pos = max(lo, min(hi, pos))
    packetHandler.write4ByteTxOnly(portHandler, ID, ADDR_GOAL_POS, int(pos) & 0xFFFFFFFF)
    if ID == 2:
        id3 = tandem_id3(pos)
        packetHandler.write4ByteTxOnly(portHandler, 3, ADDR_GOAL_POS, id3)
    return pos


def enable_torque(ID):
    packetHandler.write4ByteTxOnly(portHandler, ID, ADDR_PROF_ACC, PROFILE_ACC)
    time.sleep(0.05)
    packetHandler.write4ByteTxOnly(portHandler, ID, ADDR_VELOCITY, VELOCITY[ID])
    time.sleep(0.05)
    packetHandler.write1ByteTxOnly(portHandler, ID, ADDR_TORQUE, 1)
    time.sleep(0.05)


def disable_torque(ID):
    packetHandler.write1ByteTxOnly(portHandler, ID, ADDR_TORQUE, 0)
    time.sleep(0.05)


def run_diagnostics():
    """Ping every servo and print results to stdout before the TUI starts."""
    print(f'\nDiagnosing {len(ALL_JOINTS)} servos @ {BAUDRATE} bps on {DEVICE}')
    ok = 0
    for ID in ALL_JOINTS:
        model, r, _ = packetHandler.ping(portHandler, ID)
        if r == dxl.COMM_SUCCESS:
            print(f'  ID {ID:2d}: OK    model={model}')
            ok += 1
        else:
            print(f'  ID {ID:2d}: FAIL  {packetHandler.getTxRxResult(r)}')
    if ok == 0:
        print('  *** 0 / 7 responded — check power cable and USB ***')
    else:
        print(f'  {ok} / {len(ALL_JOINTS)} responded')
    print()
    return ok


def check_hw_errors(torque_on):
    for ID in ALL_JOINTS:
        hw, r, _ = packetHandler.read1ByteTxRx(portHandler, ID, ADDR_HW_ERROR)
        if r == dxl.COMM_SUCCESS and hw:
            names = [v for k, v in HW_ERRORS.items() if hw & k]
            logmsg(f'ID={ID} {",".join(names)} — rebooting')
            packetHandler.reboot(portHandler, ID)
            time.sleep(0.3)
            if torque_on:
                enable_torque(ID)


# ── Keyboard (raw mode, non-blocking) ─────────────────────────────────────────
def getch_nowait():
    """Return key string if pressed, else None. Handles arrow/function keys."""
    fd = sys.stdin.fileno()
    if not select.select([sys.stdin], [], [], 0)[0]:
        return None
    ch = os.read(fd, 1)
    if ch == b'\x1b':                     # escape sequence
        if select.select([sys.stdin], [], [], 0.02)[0]:
            seq = os.read(fd, 2)
            if seq == b'[A': return 'UP'
            if seq == b'[B': return 'DOWN'
            if seq == b'[C': return 'RIGHT'
            if seq == b'[D': return 'LEFT'
            if seq.startswith(b'[1') and select.select([sys.stdin], [], [], 0.02)[0]:
                extra = os.read(fd, 2)
                n = seq[1:] + extra.rstrip(b'~')
                fkeys = {b'[11': 'F1', b'[12': 'F2', b'[13': 'F3', b'[14': 'F4'}
                return fkeys.get(b'[' + n.lstrip(b'['), None)
        return 'ESC'
    return ch.decode('utf-8', errors='replace')


# ── Display ────────────────────────────────────────────────────────────────────
def draw(positions, goals, selected, step, torque_on):
    out = [CLEAR]
    torque_color = GREEN if torque_on else RED
    out.append(f'{BOLD}{CYAN}=== WidowX-200 Keyboard Control ==={RESET}\n')
    out.append(f'1-6 select  ←→ move  +/- step  H home  T torque  W waypoint  Q quit\n')
    out.append(f'Step: {BOLD}{step}{RESET} ticks    '
               f'Torque: {torque_color}{BOLD}{"ON" if torque_on else "OFF"}{RESET}\n')
    out.append(f'{"─"*54}\n')
    out.append(f'{"":3} {"ID":<3} {"Joint":<13} {"Actual":>8} {"Goal":>8}\n')
    out.append(f'{"─"*54}\n')

    for ID in ALL_JOINTS:
        is_slave    = (ID == 3)
        is_selected = (not is_slave) and (CTRL_JOINTS[selected] == ID)
        pos  = positions.get(ID)
        goal = goals.get(ID, HOME[ID])
        pos_str  = str(pos)  if pos  is not None else '---'
        goal_str = str(goal) if goal is not None else '---'

        prefix = '>>>' if is_selected else '   '
        line = f'{prefix} {ID:<3} {JOINT_NAMES[ID]} {pos_str:>8} {goal_str:>8}'

        if is_selected:
            out.append(f'{REVERSE}{line}{RESET}\n')
        elif is_slave:
            out.append(f'{DIM}{line}{RESET}\n')
        else:
            out.append(line + '\n')

    out.append(f'{"─"*54}\n')
    wp_count = len(json.load(open(WAYPOINTS_FILE))) if os.path.exists(WAYPOINTS_FILE) else 0
    out.append(f'{BOLD}Saved:{RESET}  ' +
               '  '.join(f'F{k}:{name}' for k, (name, _) in SAVED_POSES.items()) +
               f'    {BOLD}Waypoints:{RESET} {wp_count}\n')
    out.append(f'{"─"*54}\n')
    out.append(f'{BOLD}Log:{RESET}\n')
    for msg in log:
        out.append(f'  {DIM}{msg}{RESET}\n')

    sys.stdout.write(''.join(out))
    sys.stdout.flush()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    if not portHandler.openPort():
        print(f'ERROR: cannot open {DEVICE}'); return
    portHandler.setBaudRate(BAUDRATE)
    portHandler.clearPort()
    time.sleep(0.1)
    logmsg(f'Connected to {DEVICE}')

    run_diagnostics()

    # Reboot all servos to clear any error state, then re-enable
    for ID in ALL_JOINTS:
        packetHandler.reboot(portHandler, ID)
        time.sleep(0.12)
    time.sleep(1.5)
    portHandler.clearPort()
    logmsg('All servos rebooted')

    # combined goals + initial positions in one pass (saves 7 serial reads)
    goals = {}
    positions = {}
    for ID in ALL_JOINTS:
        pos = get_pos(ID)
        goals[ID] = pos if pos is not None else HOME[ID]
        positions[ID] = pos

    # if nothing responded at all, servos are stuck — reboot everything
    if all(v is None for v in positions.values()):
        logmsg('No response — rebooting all servos')
        portHandler.clearPort()
        for ID in ALL_JOINTS:
            packetHandler.reboot(portHandler, ID)
        time.sleep(1.5)
        portHandler.clearPort()
        run_diagnostics()
        for ID in ALL_JOINTS:
            pos = get_pos(ID)
            goals[ID] = pos if pos is not None else HOME[ID]
            positions[ID] = pos

    selected = 0
    step     = 50
    torque_on = True

    for ID in ALL_JOINTS:
        enable_torque(ID)

    # cbreak: disables echo + line buffering, keeps output processing intact
    fd  = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    sys.stdout.write('\033[?25l')  # hide cursor
    sys.stdout.flush()

    tick = 0
    try:
        while True:
            draw(positions, goals, selected, step, torque_on)

            key = getch_nowait()

            if key in ('q', 'Q', '\x03'):
                break
            elif key in [str(i) for i in range(1, 7)]:
                selected = int(key) - 1
            elif key == 'UP':
                selected = (selected - 1) % len(CTRL_JOINTS)
            elif key == 'DOWN':
                selected = (selected + 1) % len(CTRL_JOINTS)
            elif key == 'RIGHT':
                ID = CTRL_JOINTS[selected]
                goals[ID] = set_pos(ID, goals[ID] + step)
                if ID == 2: goals[3] = tandem_id3(goals[2])
            elif key == 'LEFT':
                ID = CTRL_JOINTS[selected]
                goals[ID] = set_pos(ID, goals[ID] - step)
                if ID == 2: goals[3] = tandem_id3(goals[2])
            elif key in ('+', '='):
                step = min(500, step + 10)
            elif key == '-':
                step = max(10, step - 10)
            elif key in ('h', 'H'):
                if not torque_on:
                    torque_on = True
                    for ID in ALL_JOINTS: enable_torque(ID)
                for ID in CTRL_JOINTS:
                    goals[ID] = set_pos(ID, HOME[ID])
                goals[3] = tandem_id3(goals[2])
                logmsg('Going home')
            elif key == 'F1':
                name, pose = SAVED_POSES[1]
                if not torque_on:
                    torque_on = True
                    for ID in ALL_JOINTS: enable_torque(ID)
                for ID in CTRL_JOINTS:
                    goals[ID] = set_pos(ID, pose[ID])
                goals[3] = tandem_id3(goals[2])
                logmsg(f'Moving to {name}')
            elif key in ('t', 'T'):
                torque_on = not torque_on
                for ID in ALL_JOINTS:
                    enable_torque(ID) if torque_on else disable_torque(ID)
                logmsg(f'Torque {"enabled" if torque_on else "disabled"}')
            elif key in ('w', 'W'):
                if any(v is not None for v in positions.values()):
                    waypoints = json.load(open(WAYPOINTS_FILE)) if os.path.exists(WAYPOINTS_FILE) else []
                    wp = {str(k): v for k, v in positions.items() if v is not None}
                    ee = _compute_ee(positions)
                    if ee is not None:
                        wp['x'] = round(float(ee[0]), 4)
                        wp['y'] = round(float(ee[1]), 4)
                        wp['z'] = round(float(ee[2]), 4)
                    waypoints.append(wp)
                    json.dump(waypoints, open(WAYPOINTS_FILE, 'w'), indent=2)
                    logmsg(f'Waypoint {len(waypoints)} saved')
                else:
                    logmsg('Cannot save — no position data')

            time.sleep(0.05)

            positions = {ID: get_pos(ID) for ID in ALL_JOINTS}
            if all(v is None for v in positions.values()):
                portHandler.clearPort()
                logmsg('Bus stall — flushed')
            if tick % 30 == 0:
                check_hw_errors(torque_on)
            tick += 1

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        sys.stdout.write('\033[?25h\033[2J\033[H')  # restore cursor, clear screen
        sys.stdout.flush()
        try:
            for ID in ALL_JOINTS:
                disable_torque(ID)
            portHandler.closePort()
        except Exception:
            pass
        print('Bye.')


if __name__ == '__main__':
    main()
