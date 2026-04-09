import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import urllib.request
import os
import sys
import argparse

# MediaPipe hand landmark connections (21 keypoints)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]

FACE_MODEL = "face_detector.tflite"
HAND_MODEL = "hand_landmarker.task"
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def download_model(path, url):
    if not os.path.exists(path):
        print(f"Downloading {os.path.basename(path)} ...")
        urllib.request.urlretrieve(url, path)
        print(f"Saved {path}")


def run_detector(source=0):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    face_model_path = os.path.join(script_dir, FACE_MODEL)
    hand_model_path = os.path.join(script_dir, HAND_MODEL)

    download_model(face_model_path, FACE_MODEL_URL)
    download_model(hand_model_path, HAND_MODEL_URL)

    face_options = vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=face_model_path),
        running_mode=vision.RunningMode.VIDEO,
        min_detection_confidence=0.5,
    )
    hand_options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=hand_model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=4,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: could not open video source '{source}'")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print("Running — press 'q' to quit.")

    with vision.FaceDetector.create_from_options(face_options) as face_detector, \
         vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker:

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int(frame_idx * (1000 / fps))
            frame_idx += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            face_result = face_detector.detect_for_video(mp_image, timestamp_ms)
            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

            h, w, _ = frame.shape

            # Draw face bounding boxes
            for detection in face_result.detections:
                bbox = detection.bounding_box
                x, y = bbox.origin_x, bbox.origin_y
                bw, bh = bbox.width, bbox.height
                score = detection.categories[0].score
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 200, 0), 2)
                cv2.putText(frame, f"Face {score:.0%}", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

            # Draw hand landmarks and connections
            for i, hand_landmarks in enumerate(hand_result.hand_landmarks):
                for start_idx, end_idx in HAND_CONNECTIONS:
                    s = hand_landmarks[start_idx]
                    e = hand_landmarks[end_idx]
                    cv2.line(frame,
                             (int(s.x * w), int(s.y * h)),
                             (int(e.x * w), int(e.y * h)),
                             (255, 200, 0), 2)

                for lm in hand_landmarks:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (255, 50, 50), -1)

                if hand_result.handedness and i < len(hand_result.handedness):
                    label = hand_result.handedness[i][0].display_name
                    wrist = hand_landmarks[0]
                    cv2.putText(frame, label,
                                (int(wrist.x * w) - 20, int(wrist.y * h) + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 50, 50), 2)

            # HUD
            face_count = len(face_result.detections)
            hand_count = len(hand_result.hand_landmarks)
            cv2.putText(frame, f"Faces: {face_count}  Hands: {hand_count}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Face & Hand Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect faces and hands in video.")
    parser.add_argument("source", nargs="?", default=0,
                        help="Video file path or webcam index (default: 0)")
    args = parser.parse_args()

    source = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    run_detector(source)
