// Diagnostic: ping all servos using DynamixelSDK on OpenCM9.04
// Reports results over USB Serial

#include <DynamixelSDK.h>

#define DXL_BAUDRATE 1000000
#define DXL_PORT     "1"      // Serial1 = DXL bus

dynamixel::PortHandler   *portHandler;
dynamixel::PacketHandler *packetHandler;

void setup() {
  Serial.begin(57600);
  delay(1000);
  Serial.println("DXL Test starting...");

  portHandler   = dynamixel::PortHandler::getPortHandler(DXL_PORT);
  packetHandler = dynamixel::PacketHandler::getPacketHandler(2.0);

  if (portHandler->openPort()) {
    Serial.println("Port opened");
  } else {
    Serial.println("FAILED to open port");
    return;
  }

  if (portHandler->setBaudRate(DXL_BAUDRATE)) {
    Serial.println("Baudrate set to 1Mbps");
  } else {
    Serial.println("FAILED to set baudrate");
    return;
  }

  for (int id = 1; id <= 7; id++) {
    uint8_t error = 0;
    uint16_t model = 0;
    int result = packetHandler->ping(portHandler, id, &model, &error);
    if (result == COMM_SUCCESS) {
      Serial.print("  ID="); Serial.print(id);
      Serial.print(" model="); Serial.println(model);
    } else {
      Serial.print("  ID="); Serial.print(id);
      Serial.print(" FAIL: "); Serial.println(packetHandler->getTxRxResult(result));
    }
  }

  Serial.println("Done.");
}

void loop() {
  delay(3000);
  Serial.println("--- repeat scan ---");
  for (int id = 1; id <= 7; id++) {
    uint8_t error = 0;
    uint16_t model = 0;
    int result = packetHandler->ping(portHandler, id, &model, &error);
    if (result == COMM_SUCCESS) {
      Serial.print("  ID="); Serial.print(id);
      Serial.print(" model="); Serial.println(model);
    } else {
      Serial.print("  ID="); Serial.print(id);
      Serial.println(" FAIL");
    }
  }
}
