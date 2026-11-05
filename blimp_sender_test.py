
import cv2
import requests
import serial
import time
import winsound

# -----------------------------------
# DRIFTBLIMP SETTINGS
# -----------------------------------

# Arduino Uno serial connection
ARDUINO_PORT = "COM8"
BAUD_RATE = 19200

# Homebase PC IP address
HOMEBASE_URL = "http://10.46.19.22:5000"

# USB camera index
CAMERA_INDEX = 0

# How frequently to check YOLO status
STATUS_INTERVAL = 0.5

# Audio alert settings
BEEP_FREQUENCY = 1500   # Hz
BEEP_DURATION = 500     # milliseconds

# -----------------------------------
# CONNECT TO ARDUINO
# -----------------------------------

arduino = serial.Serial(
    ARDUINO_PORT,
    BAUD_RATE,
    timeout=0.2
)

# Allow Arduino to restart
time.sleep(2)

print("Connected to Arduino Uno!")

# -----------------------------------
# CONNECT TO USB CAMERA
# -----------------------------------

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    arduino.close()
    raise RuntimeError("USB camera could not be opened")

print("USB camera connected!")

# -----------------------------------
# DETECTION STATUS TRACKING
# -----------------------------------

last_event_id = None
last_status_check = 0

session = requests.Session()


def check_homebase_status():
    global last_event_id

    try:
        response = session.get(
            HOMEBASE_URL + "/status",
            timeout=2
        )

        response.raise_for_status()
        data = response.json()

        event_id = int(data["event_id"])

        # First status check: establish baseline
        if last_event_id is None:
            last_event_id = event_id
            return

        # Detect a new object-found event
        if event_id > last_event_id:

            last_event_id = event_id

            print("\nTARGET OBJECT FOUND!")

            # Play sound through computer speakers
            winsound.Beep(
                BEEP_FREQUENCY,
                BEEP_DURATION
            )

            print("Audio alert played!")

        elif event_id < last_event_id:
            # Homebase may have restarted
            last_event_id = event_id

    except (
        requests.RequestException,
        ValueError,
        KeyError,
        TypeError
    ) as error:

        print("Status check failed:", error)


# -----------------------------------
# START DRIFTBLIMP SENDER
# -----------------------------------

print("\nDriftBlimp sender started!")
print("Waiting for Arduino CAPTURE commands...")
print("Press Ctrl+C to stop.\n")

try:
    while True:

        # --------------------------------
        # READ ARDUINO COMMAND
        # --------------------------------

        message = arduino.readline().decode(
            errors="ignore"
        ).strip()

        if message == "CAPTURE":

            print("\nCAPTURE command received!")

            # --------------------------------
            # CAPTURE USB CAMERA IMAGE
            # --------------------------------

            success, frame = cap.read()

            if not success:
                print("Camera capture failed")
                continue

            print("Picture captured!")

            # --------------------------------
            # CONVERT IMAGE TO JPEG
            # --------------------------------

            encoded, buffer = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 80]
            )

            if not encoded:
                print("JPEG encoding failed")
                continue

            # --------------------------------
            # TRANSMIT IMAGE TO HOMEBASE
            # --------------------------------

            try:
                response = session.post(
                    HOMEBASE_URL + "/upload",
                    data=buffer.tobytes(),
                    headers={
                        "Content-Type": "image/jpeg"
                    },
                    timeout=5
                )

                if response.status_code == 200:
                    print("Image sent to homebase!")
                else:
                    print(
                        "Upload failed:",
                        response.status_code
                    )

            except requests.RequestException as error:
                print("Transmission failed:", error)

        # --------------------------------
        # CHECK YOLO DETECTION STATUS
        # --------------------------------

        current_time = time.monotonic()

        if (
            current_time - last_status_check
            >= STATUS_INTERVAL
        ):

            last_status_check = current_time

            check_homebase_status()


except KeyboardInterrupt:
    print("\nStopping DriftBlimp sender...")

finally:

    cap.release()
    arduino.close()
    session.close()

    cv2.destroyAllWindows()

    print("DriftBlimp sender stopped.")
