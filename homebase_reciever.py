
import os
from pathlib import Path
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

if cv2 is None:
    raise RuntimeError(
        "OpenCV (cv2) is required for this receiver. "
        "Install the GUI-enabled package in the active Python environment, "
        "e.g. `pip install opencv-python`."
    )

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
target_lock = threading.Lock()

detection_status = {
    "event_id": 0,
    "target_found": False
}

target = {
    "class_id": None,
    "name": None,
    "conf": 0.5
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
        return jsonify(detection_status.copy()), 200

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

model_path = (
    Path(__file__).resolve().parent
    / "detect"
    / "train-2"
    / "weights"
    / "best.pt"
)
if not model_path.is_file():
    raise FileNotFoundError(f"Trained model not found: {model_path}")

yolo = YOLO(str(model_path))

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


@app.route("/classes", methods=["GET"])
def get_classes():
    return jsonify(available_classes), 200


@app.route("/target", methods=["POST"])
def set_target():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Expected a JSON object"), 400

    key = " ".join(str(data.get("name", "")).casefold().split())
    match = classes_by_name.get(key)
    if not match:
        return jsonify(error="Unknown class"), 400

    try:
        confidence = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        return jsonify(error="Confidence must be a number between 0 and 1"), 400

    if not 0 <= confidence <= 1:
        return jsonify(error="Confidence must be a number between 0 and 1"), 400

    with target_lock:
        target.update(
            class_id=match[0],
            name=match[1],
            conf=confidence
        )
        with status_lock:
            detection_status["target_found"] = False

    return jsonify(status="ok", target=match[1]), 200


# ----------------------------------
# 8. Start HTTP server in background
# ----------------------------------

server_thread = threading.Thread(
    target=start_server,
    daemon=True
)

server_thread.start()

print("Homebase ready to receive images")
print("Waiting for a target from the UI or POST /target")
print("Press O to toggle other objects")
print("Press Q to quit")

show_other_objects = False
target_was_found = False
previous_target = (None, None, None)
display_window_open = False


# ----------------------------------
# 9. Process incoming images
# ----------------------------------

while True:

    try:
        frame = image_queue.get(timeout=0.1)

    except Empty:
        # waitKey requires a HighGUI window, which is created by the first imshow.
        if not display_window_open:
            continue

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
    with target_lock:
        target_class_id = target["class_id"]
        target_name = target["name"]
        confidence_threshold = target["conf"]

    frame_target = (
        target_class_id,
        target_name,
        confidence_threshold
    )
    if frame_target != previous_target:
        target_was_found = False
        previous_target = frame_target

    for box in results.boxes:

        conf = float(box.conf[0])

        if conf < confidence_threshold:
            continue

        class_id = int(box.cls[0])
        is_target = (
            target_name is not None
            and class_id == target_class_id
        )

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
        target_name is not None
        and target_confidence >= confidence_threshold
    )

    target_event = False
    with target_lock:
        current_target = (
            target["class_id"],
            target["name"],
            target["conf"]
        )
        if current_target == frame_target:
            with status_lock:
                detection_status["target_found"] = target_found

                if target_found and not target_was_found:
                    detection_status["event_id"] += 1
                    target_event = True
                    print(
                        "Detection event:",
                        detection_status["event_id"]
                    )

            target_was_found = target_found

    if target_event and target_name is not None:
        print(
            f"Found {target_name} at "
            f"{target_confidence * 100:.1f}%"
        )

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
        else (
            f"Searching for {target_name}"
            if target_name is not None
            else "Waiting for target selection"
        )
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
    display_window_open = True

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
