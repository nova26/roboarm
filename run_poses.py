import dynamixel_sdk as dxl
import time
import sys
import numpy as np

DEVICE, BAUDRATE = '/dev/ttyACM0', 1000000
HOME = {1:180, 2:70, 3:4020, 4:83, 5:146, 6:2025, 7:1494}
POS1 = {1:0, 2:0, 3:4020+0, 4:2278, 5:3001, 6:2077, 7:1890}
# ID3 tandem offset: sum(ID2+ID3) is conserved at ~4090 after arm movement.
# ID3 correct values = 4090 - ID2_value  (e.g. HOME: 4090-70=4020, MAX: 4090-290=3800)
LIMITS = {1:(0,3980),2:(70,920),3:(3170,4020),4:(0,2745),5:(0,3100),6:(100,2800),7:(1490,2600)}
VELOCITY = {1:20,2:15,3:15,4:20,5:20,6:20,7:15}
JOINTS = [1,2,3,4,5,6,7]
ADDR_TORQUE,ADDR_PROF_ACC,ADDR_VELOCITY,ADDR_GOAL_POS,ADDR_PRES_POS,ADDR_HW_ERROR = 64,108,112,116,132,70

# Gentle profile acceleration (0 = max = causes overload shutdown on a loaded arm)
PROFILE_ACC = 5

# Smooth interpolation: number of steps and delay between steps
MOVE_STEPS = 40   # split each move into this many sub-goals
MOVE_DT    = 0.15  # seconds between sub-goal broadcasts (~6 s total per move)

ph = dxl.PortHandler(DEVICE)
pk = dxl.PacketHandler(2.0)

if not ph.openPort():
    print('FAILED to open port'); sys.exit(1)
ph.setBaudRate(BAUDRATE)
print('Connected')

for ID in JOINTS:
    err, r, _ = pk.read1ByteTxRx(ph, ID, ADDR_HW_ERROR)
    if r == dxl.COMM_SUCCESS and err:
        print('Rebooting ID=' + str(ID) + ' (HW err ' + str(err) + ')')
        pk.reboot(ph, ID); time.sleep(0.4)
    pk.write4ByteTxRx(ph, ID, ADDR_PROF_ACC, PROFILE_ACC)
    pk.write4ByteTxRx(ph, ID, ADDR_VELOCITY, VELOCITY[ID])
    pk.write1ByteTxRx(ph, ID, ADDR_TORQUE, 1)


def read_ticks():
    """Read current tick positions for all joints."""
    ticks = {}
    for ID in JOINTS:
        val, r, _ = pk.read4ByteTxRx(ph, ID, ADDR_PRES_POS)
        if r == dxl.COMM_SUCCESS:
            if val > 0x7FFFFFFF:
                val -= 0x100000000
            ticks[ID] = val
        else:
            ticks[ID] = HOME[ID]  # fall back to HOME if unresponsive
    return ticks


def move(pose, name):
    """
    Move all joints smoothly to pose by interpolating from current positions.
    All joints receive updated sub-goals simultaneously at each step.
    """
    print('Moving to ' + name + '...')

    start = read_ticks()

    # Clamp targets to limits
    target = {}
    for ID in JOINTS:
        lo, hi = LIMITS[ID]
        target[ID] = max(lo, min(hi, pose[ID]))

    for step in range(1, MOVE_STEPS + 1):
        alpha = step / MOVE_STEPS
        # Send interpolated goal to every joint this step
        for ID in JOINTS:
            pos = int(round(start[ID] + alpha * (target[ID] - start[ID])))
            pk.write4ByteTxRx(ph, ID, ADDR_GOAL_POS, pos)
        time.sleep(MOVE_DT)

    # Extra settle time
    time.sleep(1.0)

    # Report actual positions
    for ID in JOINTS:
        pos, r, _ = pk.read4ByteTxRx(ph, ID, ADDR_PRES_POS)
        if r == dxl.COMM_SUCCESS:
            if pos > 0x7FFFFFFF:
                pos -= 0x100000000
            err = pos - target[ID]
            print('  ID=' + str(ID) + ' target=' + str(target[ID]) +
                  '  actual=' + str(pos) + '  err=' + str(err))


move(HOME, 'HOME')
move(POS1, 'POS1')

for ID in JOINTS:
    pk.write1ByteTxRx(ph, ID, ADDR_TORQUE, 0)
ph.closePort()
print('Done')
