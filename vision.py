import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"  # cv2 の import より前に書く
import cv2
from ultralytics import YOLO  # pip install ultralytics

# 顔検出モデル（Haar Cascade）
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# 物体検出モデル（YOLOv12s ：軽量・COCOの80クラスを認識）
# 初回実行時に自動でダウンロードされます
yolo = YOLO('yolov12s.pt')
CONF_THRESHOLD = 0.5  # 信頼度がこれ未満の検出は無視

# PCのカメラを起動（0番が内蔵カメラ）
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print('カメラが見つかりませんでした')
    exit()

print('カメラ起動！ Qキーで終了します')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---------- 物体検出（YOLO） ----------
    results = yolo(frame, verbose=False)[0]
    object_count = 0

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f'{yolo.names[int(box.cls[0])]} {conf:.2f}'

        # 物体の枠は青、ラベルは枠の上に表示
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 128, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), (255, 128, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        object_count += 1

    # ---------- 顔検出（Haar Cascade） ----------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Face', (x, y + h + 25),  # 物体ラベルと重ならないよう下に表示
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 検出数を画面左上に表示
    cv2.putText(frame, f'Faces: {len(faces)}  Objects: {object_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow('Real-time Face & Object Detection', frame)

    # Qキーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースを解放
cap.release()
cv2.destroyAllWindows()
print('終了しました')
