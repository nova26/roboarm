"""
Read and restore key EEPROM registers on all WidowX-200 Dynamixel servos.

Run this once after any suspected EEPROM corruption.
Uses TxOnly writes (no response expected from this adapter).
Power-cycle the arm after running to apply EEPROM changes.

Registers restored (all EEPROM, require torque=OFF):
  Addr  8 : Baud Rate         = 3   (1 Mbps)
  Addr  9 : Return Delay Time = 0   (no delay)
  Addr 10 : Drive Mode        = 0   (normal)
  Addr 11 : Operating Mode    = 3   (Position Control)
  Addr 20 : Homing Offset     = 0   (4 bytes)
  Addr 48 : Max Position Limit= 4095 (4 bytes)
  Addr 52 : Min Position Limit= 0    (4 bytes)
"""
import dynamixel_sdk as dxl, time, sys

DEVICE   = '/dev/ttyACM0'
BAUDRATE = 1_000_000
ALL_IDS  = [1, 2, 3, 4, 5, 6, 7]

ph = dxl.PortHandler(DEVICE)
pk = dxl.PacketHandler(2.0)
if not ph.openPort():
    sys.exit('Cannot open port')
ph.setBaudRate(BAUDRATE)
print(f'Connected to {DEVICE}\n')

# Disable torque on all (EEPROM writes require torque=OFF)
for ID in ALL_IDS:
    pk.write1ByteTxOnly(ph, ID, 64, 0)   # addr 64 = Torque Enable
time.sleep(0.2)

print('── Reading current EEPROM values ──────────────────────────────')
print(f'{"ID":>3}  {"BaudRate":>9}  {"RDT":>5}  {"DriveMode":>10}  {"OpMode":>7}  {"HomingOff":>10}  {"MaxPos":>7}  {"MinPos":>7}')
for ID in ALL_IDS:
    def r1(addr):
        v, r, _ = pk.read1ByteTxRx(ph, ID, addr)
        return v if r == dxl.COMM_SUCCESS else '?'
    def r4(addr):
        v, r, _ = pk.read4ByteTxRx(ph, ID, addr)
        if r != dxl.COMM_SUCCESS: return '?'
        return v if v <= 0x7FFFFFFF else v - 0x100000000
    baud = r1(8); rdt = r1(9); drv = r1(10); mode = r1(11)
    hom  = r4(20); maxp = r4(48); minp = r4(52)
    print(f'{ID:>3}  {str(baud):>9}  {str(rdt):>5}  {str(drv):>10}  {str(mode):>7}  {str(hom):>10}  {str(maxp):>7}  {str(minp):>7}')

print()
resp = input('Reset all to safe defaults? [y/N] ').strip().lower()
if resp != 'y':
    print('Aborted.')
    ph.closePort()
    sys.exit(0)

print('\nWriting defaults...')
for ID in ALL_IDS:
    pk.write1ByteTxOnly(ph, ID,  8, 3)          # Baud Rate = 3 (1 Mbps)
    pk.write1ByteTxOnly(ph, ID,  9, 0)          # Return Delay Time = 0
    pk.write1ByteTxOnly(ph, ID, 10, 0)          # Drive Mode = normal
    pk.write1ByteTxOnly(ph, ID, 11, 3)          # Operating Mode = Position Control
    time.sleep(0.05)
    pk.write4ByteTxOnly(ph, ID, 20, 0)          # Homing Offset = 0
    pk.write4ByteTxOnly(ph, ID, 48, 4095)       # Max Position Limit = 4095
    pk.write4ByteTxOnly(ph, ID, 52, 0)          # Min Position Limit = 0
    time.sleep(0.05)
    print(f'  ID{ID}: done')

time.sleep(0.3)
ph.closePort()
print('\nDone. Power-cycle the arm now to apply the EEPROM changes.')
