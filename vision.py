import os
from pathlib import Path

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"  # cv2 の import より前に書く
import cv2
import torch
from textwrap import fill
from ultralytics import YOLO  # pip install ultralytics

# 顔検出モデル（Haar Cascade）
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# 物体検出モデル（YOLO11n：COCOの80クラスを認識）
# 初回実行時に自動でダウンロードされます
yolo = YOLO('yolo11n.pt')

if isinstance(yolo.names, dict):
    class_names = yolo.names
else:
    class_names = dict(enumerate(yolo.names))

classes_by_name = {
    ' '.join(name.casefold().split()): (class_id, name)
    for class_id, name in class_names.items()
}

available_classes = [name for _, name in sorted(class_names.items())]
print('Available YOLO classes:')
print(fill(', '.join(available_classes), width=88, subsequent_indent='  '))

while True:
    target_input = input('Item to find (enter an exact class name from the list): ').strip()
    target_match = classes_by_name.get(' '.join(target_input.casefold().split()))
    if target_match:
        target_class_id, target_name = target_match
        break
    print('That item is not a class in this model. Try an exact class name, such as "bottle".')

while True:
    threshold_input = input('Minimum confidence (0-100%, default 50): ').strip()
    try:
        confidence_threshold = float(threshold_input) / 100 if threshold_input else 0.5
    except ValueError:
        print('Enter a number from 0 to 100.')
        continue
    if 0 <= confidence_threshold <= 1:
        break
    print('Enter a number from 0 to 100.')

# PCのカメラを起動（0番が内蔵カメラ）
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print('カメラが見つかりませんでした')
    exit()

print(f'Looking for "{target_name}" at {confidence_threshold * 100:g}% confidence or higher.')
print('Press O to toggle other objects; press Q to quit.')
show_other_objects = False
target_was_found = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---------- 物体検出（YOLO） ----------
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
            target_confidence = max(target_confidence, conf)
        elif not show_other_objects:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f'{class_names[class_id]} {conf * 100:.1f}%'

        # 物体の枠は青、ラベルは枠の上に表示
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 128, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = max(y1 - th - 8, 0)
        cv2.rectangle(frame, (x1, label_y), (x1 + tw, label_y + th + 8), (255, 128, 0), -1)
        cv2.putText(frame, label, (x1, label_y + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        object_count += 1

    target_found = target_confidence >= confidence_threshold
    if target_found and not target_was_found:
        print(f'Found {target_name} at {target_confidence * 100:.1f}% confidence')
    target_was_found = target_found

    # ---------- 顔検出（Haar Cascade） ----------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Face', (x, y + h + 25),  # 物体ラベルと重ならないよう下に表示
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    target_status = (
        f'Found {target_name}: {target_confidence * 100:.1f}%'
        if target_found else f'Searching for {target_name} (min {confidence_threshold * 100:g}%)'
    )
    cv2.putText(frame, target_status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f'Faces: {len(faces)}  Objects: {object_count}  Other objects: {"ON" if show_other_objects else "OFF"}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow('Real-time Face & Object Detection', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('o'):
        show_other_objects = not show_other_objects
        print(f'Other objects: {"ON" if show_other_objects else "OFF"}')
    elif key == ord('q'):
        break

# リソースを解放
cap.release()
cv2.destroyAllWindows()
print('終了しました')
