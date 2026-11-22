import argparse
import os
from pathlib import Path
from textwrap import fill

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real-time YOLO detection using a trained model."
    )
    parser.add_argument(
        "model",
        nargs="?",
        help="Path to a trained .pt model, relative to the current directory.",
    )
    return parser.parse_args()


def get_model_path(model_input: str | None) -> Path:
    if model_input is None:
        model_input = input("Path to trained YOLO model (.pt): ").strip()

    model_input = model_input.strip().strip('"')
    model_path = Path(model_input).expanduser()
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    model_path = model_path.resolve()

    if model_path.suffix.lower() != ".pt":
        raise SystemExit(f"Model must be a .pt file: {model_path}")
    if not model_path.is_file():
        raise SystemExit(f"Trained model not found: {model_path}")

    return model_path


def choose_target(class_names: dict[int, str]) -> tuple[int, str]:
    classes_by_name = {
        " ".join(name.casefold().split()):
        (class_id, name)
        for class_id, name in class_names.items()
    }

    available_classes = [name for _, name in sorted(class_names.items())]
    print("Available YOLO classes:")
    print(fill(", ".join(available_classes), width=88, subsequent_indent="  "))

    while True:
        target_input = input(
            "Item to find (enter an exact class name from the list): "
        ).strip()
        target_match = classes_by_name.get(
            " ".join(target_input.casefold().split())
        )
        if target_match:
            return target_match
        print(
            "That item is not a class in this model. "
            'Try an exact class name, such as "bottle".'
        )


def choose_threshold() -> float:
    while True:
        threshold_input = input(
            "Minimum confidence (0-100%, default 50): "
        ).strip()
        try:
            confidence_threshold = (
                float(threshold_input) / 100 if threshold_input else 0.5
            )
        except ValueError:
            print("Enter a number from 0 to 100.")
            continue

        if 0 <= confidence_threshold <= 1:
            return confidence_threshold
        print("Enter a number from 0 to 100.")


def main() -> None:
    args = parse_args()
    model_path = get_model_path(args.model)

    yolo = YOLO(str(model_path))
    model_names = yolo.names
    class_names = (
        model_names
        if isinstance(model_names, dict)
        else dict(enumerate(model_names))
    )

    target_class_id, target_name = choose_target(class_names)
    confidence_threshold = choose_threshold()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit("Camera could not be opened")

    print(
        f'Looking for "{target_name}" at '
        f"{confidence_threshold * 100:g}% confidence or higher."
    )
    print("Press O to toggle other objects; press Q to quit.")

    show_other_objects = False
    target_was_found = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

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
                label = f"{class_names[class_id]} {conf * 100:.1f}%"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 128, 0), 2)
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                label_y = max(y1 - th - 8, 0)
                cv2.rectangle(
                    frame,
                    (x1, label_y),
                    (x1 + tw, label_y + th + 8),
                    (255, 128, 0),
                    -1,
                )
                cv2.putText(
                    frame,
                    label,
                    (x1, label_y + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                object_count += 1

            target_found = target_confidence >= confidence_threshold
            if target_found and not target_was_found:
                print(
                    f"Found {target_name} at "
                    f"{target_confidence * 100:.1f}% confidence"
                )
            target_was_found = target_found

            target_status = (
                f"Found {target_name}: {target_confidence * 100:.1f}%"
                if target_found
                else (
                    f"Searching for {target_name} "
                    f"(min {confidence_threshold * 100:g}%)"
                )
            )
            cv2.putText(
                frame,
                target_status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                f'  Objects: {object_count}  '
                f'Other objects: {"ON" if show_other_objects else "OFF"}',
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )

            cv2.imshow("Real-time Object Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("o"):
                show_other_objects = not show_other_objects
                print(
                    f'Other objects: '
                    f'{"ON" if show_other_objects else "OFF"}'
                )
            elif key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("終了しました")


if __name__ == "__main__":
    main()
