// USB to DYNAMIXEL transparent bridge for OpenCM9.04
// Uses DynamixelSDK PortHandler (handles TX/RX direction internally).
// Buffers complete DXL2.0 packets from USB before forwarding to DXL bus.

#include <DynamixelSDK.h>

#define DXL_BAUDRATE 1000000
#define MAX_PKT      256

dynamixel::PortHandler   *dxlPort;
dynamixel::PacketHandler *dxlPkt;

static uint8_t pkt_buf[MAX_PKT];
static int     pkt_len = 0;

// Returns expected total bytes in a DXL2.0 packet, -1 if incomplete, 0 if invalid
static int packet_size(const uint8_t *buf, int n) {
  if (n < 7) return -1;
  if (buf[0]==0xFF && buf[1]==0xFF && buf[2]==0xFD && buf[3]==0x00)
    return 7 + (buf[5] | (buf[6] << 8));
  return 0;
}

void setup() {
  Serial.begin(DXL_BAUDRATE);

  dxlPort = dynamixel::PortHandler::getPortHandler("1");
  dxlPkt  = dynamixel::PacketHandler::getPacketHandler(2.0);
  dxlPort->openPort();
  dxlPort->setBaudRate(DXL_BAUDRATE);
}

void loop() {
  // PC -> DXL: accumulate bytes until complete packet
  while (Serial.available() && pkt_len < MAX_PKT)
    pkt_buf[pkt_len++] = (uint8_t)Serial.read();

  if (pkt_len > 0) {
    int sz = packet_size(pkt_buf, pkt_len);
    if (sz == 0) {
      // Bad header — discard one byte and resync
      memmove(pkt_buf, pkt_buf + 1, --pkt_len);
    } else if (sz > 0 && pkt_len >= sz) {
      // Complete packet: writePort handles TX enable/disable internally
      dxlPort->writePort(pkt_buf, sz);
      memmove(pkt_buf, pkt_buf + sz, pkt_len - sz);
      pkt_len -= sz;
    }
  }

  // DXL -> PC: forward servo responses
  uint8_t tmp[64];
  int n = dxlPort->readPort(tmp, sizeof(tmp));
  if (n > 0) Serial.write(tmp, n);
}
