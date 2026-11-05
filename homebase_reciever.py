
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
import numpy as np
import threading
from queue import Queue, Empty
from textwrap import fill
from flask import Flask, request, jsonify
from ultralytics import YOLO

# ----------------------------------
# 1. Network receiver setup
# ----------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

# Shared detection status
status_lock = threading.Lock()

detection_status = {
    "event_id": 0,
    "target_found": False
}

# Keep only the newest unprocessed image
image_queue = Queue(maxsize=1)


# ----------------------------------
# 2. Receive images from blimp
# ----------------------------------

@app.route("/upload", methods=["POST"])
def receive_image():

    if request.mimetype != "image/jpeg":
        return jsonify(error="Expected JPEG"), 400

    image_data = request.get_data()

    image_array = np.frombuffer(
        image_data, dtype=np.uint8
    )

    frame = cv2.imdecode(
        image_array, cv2.IMREAD_COLOR
    )

    if frame is None:
        return jsonify(error="Invalid image"), 400

    # Discard previous image if necessary
    if image_queue.full():
        try:
            image_queue.get_nowait()
        except Empty:
            pass

    image_queue.put_nowait(frame)

    print("Image received from blimp!")

    return jsonify(status="received"), 200


# ----------------------------------
# 3. Return detection status to sender
# ----------------------------------

@app.route("/status", methods=["GET"])
def get_status():

    with status_lock:
        return jsonify(detection_status), 200


# ----------------------------------
# 4. Start Flask server
# ----------------------------------

def start_server():
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False
    )


# ----------------------------------
# 5. Load object and face detection
# ----------------------------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

yolo = YOLO("yolo11n.pt")

class_names = (
    yolo.names
    if isinstance(yolo.names, dict)
    else dict(enumerate(yolo.names))
)

classes_by_name = {
    " ".join(name.casefold().split()):
        (class_id, name)
    for class_id, name in class_names.items()
}

available_classes = [
    name for _, name in sorted(class_names.items())
]

print("Available YOLO classes:")
print(fill(", ".join(available_classes), width=88))


# ----------------------------------
# 6. Choose target object
# ----------------------------------

while True:
    target_input = input(
        "Item to find: "
    ).strip()

    target_match = classes_by_name.get(
        " ".join(target_input.casefold().split())
    )

    if target_match:
        target_class_id, target_name = target_match
        break

    print("Invalid class name. Try bottle, cup, etc.")


# ----------------------------------
# 7. Set confidence threshold
# ----------------------------------

while True:
    threshold_input = input(
        "Minimum confidence (0-100%, default 50): "
    ).strip()

    try:
        confidence_threshold = (
            float(threshold_input) / 100
            if threshold_input else 0.5
        )
    except ValueError:
        print("Enter a valid number.")
        continue

    if 0 <= confidence_threshold <= 1:
        break

    print("Enter a number between 0 and 100.")


# ----------------------------------
# 8. Start HTTP server in background
# ----------------------------------

server_thread = threading.Thread(
    target=start_server,
    daemon=True
)

server_thread.start()

print("Homebase ready to receive images")
print(f"Looking for: {target_name}")
print("Press O to toggle other objects")
print("Press Q to quit")

show_other_objects = False
target_was_found = False


# ----------------------------------
# 9. Process incoming images
# ----------------------------------

while True:

    try:
        frame = image_queue.get(timeout=0.1)

    except Empty:

        # Keep window responsive while waiting
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("o"):
            show_other_objects = not show_other_objects
            print("Other objects:", show_other_objects)

        continue

    # ----------------------------------
    # 10. YOLO object detection
    # ----------------------------------

    results = yolo(frame, verbose=False)[0]

    object_count = 0
    target_confidence = 0.0

    for box in results.boxes:

        conf = float(box.conf[0])

        if conf < confidence_threshold:
            continue

        class_id = int(box.cls[0])
        is_target = class_id == target_class_id

        if is_target:
            target_confidence = max(
                target_confidence, conf
            )

        elif not show_other_objects:
            continue

        x1, y1, x2, y2 = map(
            int, box.xyxy[0]
        )

        label = (
            f"{class_names[class_id]} "
            f"{conf * 100:.1f}%"
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 128, 0),
            2
        )

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 128, 0),
            2
        )

        object_count += 1

    # ----------------------------------
    # 11. Target identification
    # ----------------------------------

    target_found = (
        target_confidence >= confidence_threshold
    )

    # Update shared status for Computer 1
    with status_lock:

        detection_status["target_found"] = target_found

        # Increment event only when target
        # changes from NOT FOUND to FOUND
        if target_found and not target_was_found:

            detection_status["event_id"] += 1

            print(
                "Detection event:",
                detection_status["event_id"]
            )

    # Print detection message only once
    # per new target detection
    if target_found and not target_was_found:

        print(
            f"Found {target_name} at "
            f"{target_confidence * 100:.1f}%"
        )

    target_was_found = target_found

    # ----------------------------------
    # 12. Face detection
    # ----------------------------------

    gray = cv2.cvtColor(
        frame, cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Face",
            (x, y + h + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # ----------------------------------
    # 13. Display detection information
    # ----------------------------------

    status = (
        f"Found {target_name}: "
        f"{target_confidence * 100:.1f}%"
        if target_found
        else f"Searching for {target_name}"
    )

    cv2.putText(
        frame,
        status,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        f"Faces: {len(faces)} Objects: {object_count}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.imshow(
        "DriftBlimp Homebase Detection",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("o"):
        show_other_objects = not show_other_objects
        print("Other objects:", show_other_objects)

    elif key == ord("q"):
        break


# ----------------------------------
# 14. Clean up
# ----------------------------------

cv2.destroyAllWindows()
print("DriftBlimp homebase stopped")
