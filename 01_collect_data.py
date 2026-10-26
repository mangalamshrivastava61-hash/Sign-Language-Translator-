import os
import sys
import subprocess

# Auto-redirect to .venv Python if executed outside virtual environment
_base_dir = os.path.dirname(os.path.abspath(__file__))
_venv_py = os.path.join(_base_dir, ".venv", "Scripts", "python.exe")
if os.path.exists(_venv_py) and sys.executable.lower() != os.path.abspath(_venv_py).lower():
    sys.exit(subprocess.call([_venv_py] + sys.argv))

# Ensure Windows loads MediaPipe C++ framework DLLs
for path in [
    os.path.join(_base_dir, ".venv", "Lib", "site-packages", "mediapipe", "python"),
    os.path.join(_base_dir, ".venv", "Scripts")
]:
    if os.path.exists(path) and hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(path)
        except Exception:
            pass

import cv2
import time
import pickle
import numpy as np
import mediapipe as mp
from utils import extract_two_hands_features

DATA_DIR = os.path.join(_base_dir, "data")
DATA_FILE = os.path.join(DATA_DIR, "gestures_data.pickle")
os.makedirs(DATA_DIR, exist_ok=True)

# Load existing dataset if available
if os.path.exists(DATA_FILE):
    try:
        with open(DATA_FILE, 'rb') as f:
            dataset = pickle.load(f)
            all_features = dataset.get('features', [])
            all_labels = dataset.get('labels', [])
            saved_classes = list(dict.fromkeys(all_labels))
    except Exception:
        all_features, all_labels, saved_classes = [], [], []
else:
    all_features, all_labels, saved_classes = [], [], []

# Initialize MediaPipe Hands for UP TO 2 HANDS
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Support 1 or 2 hands
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

SAMPLES_PER_GESTURE = 60

print("=" * 65)
print("TWO-HAND SIGN LANGUAGE DATASET RECORDER")
print("Supports single-hand AND two-hand signs!")
print("=" * 65)
if saved_classes:
    print(f"Current words in dataset ({len(saved_classes)}): {saved_classes}")
else:
    print("Dataset is currently empty.")
print("=" * 65)

while True:
    print("\n" + "-" * 60)
    word_to_record = input("Enter the word/sign to record (or type 'done' to finish): ").strip()
    
    if not word_to_record or word_to_record.lower() == 'done':
        break

    if word_to_record in saved_classes:
        choice = input(f"'{word_to_record}' already exists! Overwrite it? (y/n, default: y): ").strip().lower()
        if choice != 'n':
            new_feats = [f for f, l in zip(all_features, all_labels) if l != word_to_record]
            new_lbls = [l for l in all_labels if l != word_to_record]
            all_features = new_feats
            all_labels = new_lbls
            if word_to_record in saved_classes:
                saved_classes.remove(word_to_record)

    print(f"\nOpening camera for: '{word_to_record}'...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam. Please check camera permissions.")
        continue

    current_samples = []
    is_capturing = False

    print("Camera open! Position your hand(s) and [HOLD SPACEBAR] to record.")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            num_detected = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

            # Draw skeletons for all detected hands
            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Color-code Left (cyan) vs Right (magenta/green)
                    skeleton_color = (0, 255, 255) if idx == 0 else (255, 0, 255)
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=skeleton_color, thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(0, 200, 0) if is_capturing else skeleton_color, thickness=2)
                    )

            # Record samples when SPACEBAR is held
            if is_capturing:
                if num_detected > 0:
                    features_126 = extract_two_hands_features(results)
                    current_samples.append((features_126, word_to_record))
                    cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 0), 6)
                else:
                    cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 6)
                    cv2.putText(frame, "NO HANDS IN VIEW!", (w // 2 - 170, h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                if len(current_samples) >= SAMPLES_PER_GESTURE:
                    for feat, lbl in current_samples:
                        all_features.append(feat)
                        all_labels.append(lbl)
                    if word_to_record not in saved_classes:
                        saved_classes.append(word_to_record)
                    print(f"Captured {len(current_samples)} samples for '{word_to_record}'!")
                    break

            # Top UI banner
            cv2.rectangle(frame, (0, 0), (w, 85), (25, 25, 25), -1)
            cv2.putText(frame, f"Recording: {word_to_record}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            
            hands_info = f"Hands detected: {num_detected} | Samples: {len(current_samples)}/{SAMPLES_PER_GESTURE}"
            status_color = (0, 255, 0) if is_capturing and num_detected > 0 else (200, 200, 200)
            cv2.putText(frame, hands_info, (20, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

            cv2.putText(frame, "[HOLD SPACEBAR TO RECORD | Q: Cancel]", (w - 460, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1)

            # Progress bar
            bar_width = int((len(current_samples) / SAMPLES_PER_GESTURE) * (w - 40))
            cv2.rectangle(frame, (20, h - 35), (w - 20, h - 15), (60, 60, 60), 2)
            cv2.rectangle(frame, (20, h - 35), (20 + bar_width, h - 15), (0, 255, 0), -1)

            cv2.imshow(f"Record Sign - {word_to_record} (2-Hand Mode)", frame)

            key = cv2.waitKey(20) & 0xFF
            if key == 32:  # Spacebar
                is_capturing = True
            else:
                is_capturing = False

            if key in (ord('q'), ord('Q')):
                print("Cancelled recording for this word.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    with open(DATA_FILE, 'wb') as f:
        pickle.dump({'features': all_features, 'labels': all_labels, 'classes': saved_classes}, f)
    print(f"Dataset updated! Total words in dataset: {saved_classes}")

hands.close()
print("\n" + "=" * 65)
print(f"DATASET SAVED! Total {len(all_features)} samples across {len(saved_classes)} words: {saved_classes}")
print("Feature dimension: 126 (Supports 1 and 2 hands)")
print("Now run 'python 02_train_model.py' to train your model!")
print("=" * 65)
