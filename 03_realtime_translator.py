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
import threading
import numpy as np
import mediapipe as mp
from utils import normalize_single_hand, extract_two_hands_features

# Optional Text-to-Speech
try:
    import pyttsx3
    def speak_text(text):
        def _speak():
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.say(text)
                engine.runAndWait()
            except Exception:
                pass
        threading.Thread(target=_speak, daemon=True).start()
except ImportError:
    def speak_text(text):
        print(f"[TTS Audio]: {text}")

MODEL_FILE = os.path.join(_base_dir, "model", "sign_classifier.p")

# ==========================================
# PAUSE / COOLDOWN CONFIGURATION
# ==========================================
PAUSE_BETWEEN_GESTURES = 1.0  # Pause duration (seconds) between signs
CONFIDENCE_THRESHOLD = 0.60   # Minimum prediction certainty (60%)
HOLD_FRAMES_TO_COMMIT = 15    # ~0.5 to 0.7s of steady holding to auto-add word

if not os.path.exists(MODEL_FILE):
    print(f"Error: Model file '{MODEL_FILE}' not found! Please run '02_train_model.py' first.")
    exit(1)

with open(MODEL_FILE, 'rb') as f:
    model_data = pickle.load(f)

clf = model_data['model']
classes = model_data['classes']
EXPECTED_FEATURES = getattr(clf, 'n_features_in_', 63)

print("=" * 65)
print("REAL-TIME SIGN LANGUAGE TRANSLATOR")
print(f"Loaded Model Mode: {'Two-Hand (126 features)' if EXPECTED_FEATURES == 126 else 'Single-Hand (63 features)'}")
print(f"Recognizable signs ({len(classes)}): {list(classes)}")
print(f"Pause between gestures: {PAUSE_BETWEEN_GESTURES}s")
print("Controls:")
print(" - [SPACE]: Commit current sign to sentence")
print(" - [BACKSPACE] or 'b': Delete last word")
print(" - [C]: Clear sentence")
print(" - [S]: Speak sentence aloud (TTS)")
print(" - [Q]: Quit")
print("=" * 65)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2 if EXPECTED_FEATURES == 126 else 1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

sentence = []
current_stable_sign = None
stable_frame_count = 0
last_committed_time = 0.0
currently_seen_sign = None

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        now = time.time()

        # Cooldown pause timer
        time_since_commit = now - last_committed_time
        in_cooldown = time_since_commit < PAUSE_BETWEEN_GESTURES
        remaining_pause = max(0.0, PAUSE_BETWEEN_GESTURES - time_since_commit)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        detected_sign = None
        confidence = 0.0
        num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

        if num_hands > 0:
            all_x, all_y = [], []
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                skeleton_color = (0, 140, 255) if in_cooldown else ((0, 255, 255) if idx == 0 else (255, 0, 255))
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=skeleton_color, thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=skeleton_color, thickness=2)
                )
                for lm in hand_landmarks.landmark:
                    all_x.append(int(lm.x * w))
                    all_y.append(int(lm.y * h))

            # Dynamically adapt feature vector to match model expectations
            if not in_cooldown:
                if EXPECTED_FEATURES == 126:
                    features = extract_two_hands_features(results)
                else:
                    features = normalize_single_hand(results.multi_hand_landmarks[0])

                features = np.expand_dims(features, axis=0)

                probs = clf.predict_proba(features)[0]
                best_idx = np.argmax(probs)
                confidence = probs[best_idx]
                candidate_sign = clf.classes_[best_idx]

                if confidence >= CONFIDENCE_THRESHOLD:
                    detected_sign = candidate_sign
                    currently_seen_sign = candidate_sign
            else:
                currently_seen_sign = None

            # Bounding box
            if all_x and all_y:
                x_min, x_max = max(0, min(all_x) - 25), min(w, max(all_x) + 25)
                y_min, y_max = max(0, min(all_y) - 25), min(h, max(all_y) + 25)

                if in_cooldown:
                    box_color = (0, 140, 255)
                    tag = f"Paused ({remaining_pause:.1f}s) - Transition hand"
                elif detected_sign is not None:
                    box_color = (0, 255, 0)
                    tag = f"{detected_sign} ({confidence*100:.0f}%)"
                else:
                    box_color = (120, 120, 120)
                    tag = "Analyzing..."

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
                cv2.rectangle(frame, (x_min, max(0, y_min - 32)), (x_min + len(tag) * 12 + 10, y_min), box_color, -1)
                cv2.putText(frame, tag, (x_min + 5, y_min - 9),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2)
        else:
            currently_seen_sign = None

        # Auto-hold commit logic
        if not in_cooldown and detected_sign is not None:
            if detected_sign == current_stable_sign:
                stable_frame_count += 1
            else:
                current_stable_sign = detected_sign
                stable_frame_count = 1

            if stable_frame_count >= HOLD_FRAMES_TO_COMMIT:
                sentence.append(detected_sign)
                last_committed_time = time.time()
                speak_text(detected_sign)
                stable_frame_count = 0
                current_stable_sign = None
        else:
            if not in_cooldown:
                stable_frame_count = max(0, stable_frame_count - 1)
                if stable_frame_count == 0:
                    current_stable_sign = None

        # --- UI OVERLAYS ---
        # 1. Top Header Bar
        cv2.rectangle(frame, (0, 0), (w, 55), (25, 25, 25), -1)
        cv2.putText(frame, "Sign Language Translator", (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(frame, "[SPACE: Add | B: Del | C: Clear | S: Speak | Q: Quit]", (w - 560, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1)

        # 2. Status / Cooldown Indicator
        if in_cooldown:
            cooldown_ratio = remaining_pause / PAUSE_BETWEEN_GESTURES
            bar_w = int(cooldown_ratio * 200)
            cv2.rectangle(frame, (20, 70), (220, 90), (50, 50, 50), 2)
            cv2.rectangle(frame, (20, 70), (20 + bar_w, 90), (0, 140, 255), -1)
            cv2.putText(frame, f"Pause: {remaining_pause:.1f}s (Prepare next sign)", (230, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
        elif current_stable_sign and stable_frame_count > 0:
            commit_ratio = min(1.0, stable_frame_count / HOLD_FRAMES_TO_COMMIT)
            bar_w = int(commit_ratio * 180)
            cv2.rectangle(frame, (20, 70), (200, 90), (60, 60, 60), 2)
            cv2.rectangle(frame, (20, 70), (20 + bar_w, 90), (0, 255, 0), -1)
            cv2.putText(frame, f"Adding: {current_stable_sign}", (215, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        else:
            cv2.circle(frame, (35, 80), 8, (0, 255, 0), -1)
            cv2.putText(frame, "Ready for next sign", (55, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # 3. Bottom Sentence Ribbon
        full_sentence = " ".join(sentence) if sentence else "(Perform sign and hold or press SPACE to add...)"
        cv2.rectangle(frame, (0, h - 70), (w, h), (15, 15, 15), -1)
        cv2.rectangle(frame, (0, h - 70), (w, h - 68), (0, 255, 255), -1)
        cv2.putText(frame, "Sentence:", (20, h - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2)
        cv2.putText(frame, full_sentence, (150, h - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Sign Language Translator", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
        elif key == 32:  # SPACE
            if not in_cooldown and currently_seen_sign:
                sentence.append(currently_seen_sign)
                last_committed_time = time.time()
                speak_text(currently_seen_sign)
                stable_frame_count = 0
                current_stable_sign = None
        elif key in (ord('c'), ord('C')):
            sentence.clear()
            stable_frame_count = 0
            current_stable_sign = None
        elif key in (ord('b'), ord('B'), 8):  # Backspace
            if sentence:
                sentence.pop()
        elif key in (ord('s'), ord('S')):
            if sentence:
                speak_text(" ".join(sentence))

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
