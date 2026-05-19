"""
Waist range explorer.
Moves the waist joint incrementally in one direction.
Press ENTER to take the next step, 's' to save current position as a limit, 'q' to quit.

Usage:
    python explore_waist.py --dir min   # move toward 0
    python explore_waist.py --dir max   # move toward 3980
"""

import sys, tty, termios, argparse, time
import dynamixel_sdk as dxl

DEVICE   = '/dev/ttyACM0'
BAUDRATE = 1_000_000
STEP     = 50   # ticks per move

ADDR_TORQUE   = 64
ADDR_PROF_ACC = 108
ADDR_VELOCITY = 112
ADDR_GOAL_POS = 116
ADDR_PRES_POS = 132

WAIST_ID  = 1
WAIST_MIN = 0
WAIST_MAX = 3980

def getch():
    fd  = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

def read_pos(ph, pk, dxl_id):
    val, r, _ = pk.read4ByteTxRx(ph, dxl_id, ADDR_PRES_POS)
    if r != dxl.COMM_SUCCESS:
        return None
    return val if val <= 0x7FFF_FFFF else val - 0x1_0000_0000

def write_pos(ph, pk, dxl_id, pos):
    pk.write4ByteTxRx(ph, dxl_id, ADDR_GOAL_POS, int(pos))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', choices=['min', 'max'], required=True,
                        help='Direction to explore: min (toward 0) or max (toward 3980)')
    parser.add_argument('--step', type=int, default=STEP)
    args = parser.parse_args()

    ph = dxl.PortHandler(DEVICE)
    pk = dxl.PacketHandler(2.0)
    if not ph.openPort():
        print(f'ERROR: cannot open {DEVICE}'); return
    ph.setBaudRate(BAUDRATE)
    ph.clearPort()

    pk.write4ByteTxRx(ph, WAIST_ID, ADDR_PROF_ACC, 3)
    pk.write4ByteTxRx(ph, WAIST_ID, ADDR_VELOCITY, 10)   # slow
    pk.write1ByteTxRx(ph, WAIST_ID, ADDR_TORQUE, 1)

    pos = read_pos(ph, pk, WAIST_ID)
    if pos is None:
        print('Cannot read waist position — check connection'); return

    direction = -args.step if args.dir == 'min' else +args.step
    limit     = WAIST_MIN if args.dir == 'min' else WAIST_MAX

    print(f'\nWaist explorer — moving toward {"MIN (0)" if args.dir == "min" else "MAX (3980)"}')
    print(f'Current position: {pos}')
    print(f'Step size: {args.step} ticks  (~{args.step * 360/4096:.1f}°)')
    print()
    print('  ENTER  — move one step')
    print('  s      — save current position as safe limit and quit')
    print('  q      — quit without saving')
    print()

    saved_limit = None
    while True:
        key = getch()

        if key in ('\r', '\n', ' '):
            next_pos = pos + direction
            next_pos = max(WAIST_MIN, min(WAIST_MAX, next_pos))
            write_pos(ph, pk, WAIST_ID, next_pos)
            time.sleep(0.8)
            pos = read_pos(ph, pk, WAIST_ID) or next_pos
            pct = pos / WAIST_MAX * 100
            print(f'  Waist: {pos:5d} ticks  ({pct:.1f}% of range)'
                  f'{"  [at limit]" if pos <= WAIST_MIN or pos >= WAIST_MAX else ""}')

            if pos == limit:
                print(f'  Reached hardware limit ({limit}).')
                break

        elif key in ('s', 'S'):
            saved_limit = pos
            print(f'\n  ✓ Saved safe {"MIN" if args.dir == "min" else "MAX"} = {saved_limit}')
            break

        elif key in ('q', 'Q', '\x03'):
            print('\n  Quit.')
            break

    pk.write1ByteTxRx(ph, WAIST_ID, ADDR_TORQUE, 0)
    ph.closePort()

    if saved_limit is not None:
        key = 'min' if args.dir == 'min' else 'max'
        print(f'\nUpdate LIMITS[1] in keyboard_control.py:')
        print(f'  {key} = {saved_limit}')
        print(f'\nUpdate the waist sweep in widowx_run.py waypoints to stay within this limit.')

if __name__ == '__main__':
    main()
