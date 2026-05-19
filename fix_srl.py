"""
One-shot: set Status Return Level=2 on all servos and verify reads work.
Run while arm is powered on, BEFORE widowx_run.py.
This fixes the "no status packet" read failures.
"""
import dynamixel_sdk as dxl, time, sys

DEVICE   = '/dev/ttyACM0'
BAUDRATE = 1_000_000
ADDR_TORQUE = 64
ADDR_SRL    = 17   # Status Return Level (EEPROM)
ALL_IDS     = [1, 2, 3, 4, 5, 6, 7]

ph = dxl.PortHandler(DEVICE)
pk = dxl.PacketHandler(2.0)
if not ph.openPort():
    sys.exit('Cannot open port')
ph.setBaudRate(BAUDRATE)

# Disable torque (required to write EEPROM)
for ID in ALL_IDS:
    pk.write1ByteTxRx(ph, ID, ADDR_TORQUE, 0)
time.sleep(0.1)

# Write SRL=2 — servo processes write even if it can't respond yet
print('Setting SRL=2 on all servos...')
for ID in ALL_IDS:
    pk.write1ByteTxRx(ph, ID, ADDR_SRL, 2)
time.sleep(0.3)
ph.ser.flushInput()

# Verify
print('Verifying:')
ok_count = 0
for ID in ALL_IDS:
    srl, r, _ = pk.read1ByteTxRx(ph, ID, ADDR_SRL)
    pos, rp,_ = pk.read4ByteTxRx(ph, ID, 132)
    if pos > 0x7FFFFFFF: pos -= 0x100000000
    status = 'OK' if r == dxl.COMM_SUCCESS else 'FAIL'
    print(f'  ID{ID}: SRL={srl} ({status})  pos={pos}')
    if r == dxl.COMM_SUCCESS:
        ok_count += 1

ph.closePort()
print(f'\n{ok_count}/{len(ALL_IDS)} reads OK')
if ok_count == len(ALL_IDS):
    print('Done — now run:  python widowx_run.py --log')
else:
    print('Some reads still failing. Run this script again while arm is powered on.')
