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
from utils import extract_and_normalize_landmarks

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
CONFIDENCE_THRESHOLD = 0.65
COMMIT_FRAMES_REQUIRED = 16
COOLDOWN_SECONDS = 0.8

if not os.path.exists(MODEL_FILE):
    print(f"Error: Model file '{MODEL_FILE}' not found! Please run '02_train_model.py' first.")
    exit(1)

with open(MODEL_FILE, 'rb') as f:
    model_data = pickle.load(f)

clf = model_data['model']
classes = model_data['classes']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
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
last_committed_sign = None
last_committed_time = 0
currently_seen_sign = None

print("=" * 65)
print("REAL-TIME SIGN LANGUAGE TRANSLATOR")
print(f"Recognizable words: {list(classes)}")
print("How to build your sentence:")
print(" - Show hand sign in camera.")
print(" - Press [SPACE] to instantly add the current word to your sentence,")
print("   OR hold the sign steady to auto-add it.")
print("Controls:")
print(" - [SPACE]: Add current word to sentence")
print(" - [BACKSPACE] or 'b': Delete last word")
print(" - [c]: Clear entire sentence")
print(" - [s]: Speak sentence aloud (TTS)")
print(" - [q]: Quit")
print("=" * 65)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        detected_sign = None
        confidence = 0.0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 165, 255), thickness=2)
                )

                features = extract_and_normalize_landmarks(hand_landmarks)
                features = np.expand_dims(features, axis=0)

                probs = clf.predict_proba(features)[0]
                best_idx = np.argmax(probs)
                confidence = probs[best_idx]
                candidate_sign = clf.classes_[best_idx]

                if confidence >= CONFIDENCE_THRESHOLD:
                    detected_sign = candidate_sign
                    currently_seen_sign = candidate_sign

                x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
                y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
                x_min, x_max = max(0, min(x_coords) - 25), min(w, max(x_coords) + 25)
                y_min, y_max = max(0, min(y_coords) - 25), min(h, max(y_coords) + 25)

                is_recognized = (detected_sign is not None)
                box_color = (0, 255, 0) if is_recognized else (0, 0, 255)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
                tag = f"{candidate_sign} ({confidence*100:.0f}%)" if is_recognized else "Unknown"
                cv2.rectangle(frame, (x_min, max(0, y_min - 32)), (x_min + len(tag) * 14 + 10, y_min), box_color, -1)
                cv2.putText(frame, tag, (x_min + 5, y_min - 9),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2)
        else:
            currently_seen_sign = None

        now = time.time()
        if detected_sign is not None:
            if detected_sign == current_stable_sign:
                stable_frame_count += 1
            else:
                current_stable_sign = detected_sign
                stable_frame_count = 1

            if stable_frame_count >= COMMIT_FRAMES_REQUIRED:
                if detected_sign != last_committed_sign or (now - last_committed_time) > COOLDOWN_SECONDS:
                    sentence.append(detected_sign)
                    last_committed_sign = detected_sign
                    last_committed_time = now
                    speak_text(detected_sign)
                    stable_frame_count = 0
        else:
            stable_frame_count = max(0, stable_frame_count - 1)
            if stable_frame_count == 0:
                current_stable_sign = None

        # Top Header Bar
        cv2.rectangle(frame, (0, 0), (w, 55), (25, 25, 25), -1)
        cv2.putText(frame, "Sign Language Translator", (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "[SPACE: Add Word | B: Delete | C: Clear | S: Speak | Q: Quit]", (w - 620, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1)

        # Hold progress bar
        if current_stable_sign and stable_frame_count > 0:
            commit_ratio = min(1.0, stable_frame_count / COMMIT_FRAMES_REQUIRED)
            progress_w = int(commit_ratio * 180)
            cv2.rectangle(frame, (20, 70), (200, 88), (60, 60, 60), 2)
            cv2.rectangle(frame, (20, 70), (20 + progress_w, 88), (0, 255, 0), -1)
            cv2.putText(frame, f"Adding: {current_stable_sign}", (215, 84),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # Bottom Sentence Banner
        full_sentence = " ".join(sentence) if sentence else "(Perform sign and press SPACE or hold to add...)"
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
            if currently_seen_sign:
                sentence.append(currently_seen_sign)
                last_committed_sign = currently_seen_sign
                speak_text(currently_seen_sign)
                stable_frame_count = 0
        elif key in (ord('c'), ord('C')):
            sentence.clear()
            last_committed_sign = None
        elif key in (ord('b'), ord('B'), 8):  # Backspace
            if sentence:
                sentence.pop()
                last_committed_sign = None
        elif key in (ord('s'), ord('S')):
            if sentence:
                speak_text(" ".join(sentence))

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
