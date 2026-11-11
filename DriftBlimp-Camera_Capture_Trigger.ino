
/*
 * DriftBlimp - Camera Capture Trigger
 * Board: Arduino Uno R3
 *
 * Purpose:
 * Send a command every ?? seconds to tell
 * the connected PC to capture an image.
 */

// CHANGE THIS to adjust capture frequency
const unsigned long CAPTURE_INTERVAL = 1000;

unsigned long lastCapture = 0;

void setup() {

    // Serial communication with PC
    Serial.begin(19200);

    // Built-in LED indicates capture request
    pinMode(LED_BUILTIN, OUTPUT);

    Serial.println("DriftBlimp Ready");
}

void loop() {

    unsigned long currentTime = millis();

    if (currentTime - lastCapture >= CAPTURE_INTERVAL) {

        lastCapture = currentTime;

        // Tell Python to capture an image
        Serial.println("CAPTURE");

        // Blink LED when command is sent
        digitalWrite(LED_BUILTIN, HIGH);
        delay(100);
        digitalWrite(LED_BUILTIN, LOW);
    }
}
