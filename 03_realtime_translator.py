import core.bootstrap                                 # venv redirect + DLL setup
import os, cv2, time, pickle, threading
import numpy as np
import mediapipe as mp
from core.config import (MODEL_FILE, CONFIDENCE_THRESHOLD, PAUSE_BETWEEN_GESTURES,
                         HOLD_FRAMES_TO_COMMIT, MP_DETECTION_CONFIDENCE,
                         MP_TRACKING_CONFIDENCE, TTS_RATE)
from core.utils import extract_two_hands_features, normalize_single_hand
from core.ui   import draw_translator_ui, draw_bounding_box

# ── Optional TTS ──────────────────────────────────────────────────────────────
try:
    import pyttsx3
    def speak(text):
        def _run():
            try:
                eng = pyttsx3.init()
                eng.setProperty('rate', TTS_RATE)
                eng.say(text)
                eng.runAndWait()
            except Exception:
                pass
        threading.Thread(target=_run, daemon=True).start()
except ImportError:
    def speak(text):
        print(f"[TTS]: {text}")

# ── Load model ────────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_FILE):
    print(f"Error: '{MODEL_FILE}' not found. Run '02_train_model.py' first."); exit(1)

with open(MODEL_FILE, 'rb') as f:
    data = pickle.load(f)

clf             = data['model']
classes         = data['classes']
EXPECTED_FEATS  = getattr(clf, 'n_features_in_', 63)
two_hand_mode   = EXPECTED_FEATS == 126

print("=" * 65)
print("REAL-TIME SIGN LANGUAGE TRANSLATOR")
print(f"Mode : {'Two-Hand (126D)' if two_hand_mode else 'Single-Hand (63D)'}")
print(f"Signs: {list(classes)}")
print(f"Controls: SPACE add | B delete | C clear | S speak | Q quit")
print("=" * 65)

# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2 if two_hand_mode else 1,
    min_detection_confidence=MP_DETECTION_CONFIDENCE,
    min_tracking_confidence=MP_TRACKING_CONFIDENCE,
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam."); exit(1)

# ── State ─────────────────────────────────────────────────────────────────────
sentence            = []
current_stable_sign = None
stable_frame_count  = 0
last_committed_time = 0.0
currently_seen_sign = None

# ── Main loop ─────────────────────────────────────────────────────────────────
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        now   = time.time()

        time_since_commit = now - last_committed_time
        in_cooldown       = time_since_commit < PAUSE_BETWEEN_GESTURES
        remaining_pause   = max(0.0, PAUSE_BETWEEN_GESTURES - time_since_commit)

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        detected_sign = None
        confidence    = 0.0
        num_hands     = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
        all_x, all_y  = [], []

        if num_hands > 0:
            for idx, lm in enumerate(results.multi_hand_landmarks):
                color = (0, 140, 255) if in_cooldown else ((0, 255, 255) if idx == 0 else (255, 0, 255))
                mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=color, thickness=2))
                for p in lm.landmark:
                    all_x.append(int(p.x * w))
                    all_y.append(int(p.y * h))

            # Prediction (skip during cooldown)
            if not in_cooldown:
                feats = extract_two_hands_features(results) if two_hand_mode else normalize_single_hand(results.multi_hand_landmarks[0])
                probs = clf.predict_proba(np.expand_dims(feats, 0))[0]
                best  = np.argmax(probs)
                confidence = probs[best]
                if confidence >= CONFIDENCE_THRESHOLD:
                    detected_sign       = clf.classes_[best]
                    currently_seen_sign = detected_sign
            else:
                currently_seen_sign = None

            draw_bounding_box(frame, all_x, all_y, w, h, in_cooldown, detected_sign, confidence, remaining_pause)
        else:
            currently_seen_sign = None

        # ── Hold-to-commit logic ──────────────────────────────────────────────
        if not in_cooldown and detected_sign:
            if detected_sign == current_stable_sign:
                stable_frame_count += 1
            else:
                current_stable_sign = detected_sign
                stable_frame_count  = 1

            if stable_frame_count >= HOLD_FRAMES_TO_COMMIT:
                sentence.append(detected_sign)
                last_committed_time = time.time()
                speak(detected_sign)
                stable_frame_count  = 0
                current_stable_sign = None
        else:
            if not in_cooldown:
                stable_frame_count = max(0, stable_frame_count - 1)
                if stable_frame_count == 0:
                    current_stable_sign = None

        # ── Draw UI ───────────────────────────────────────────────────────────
        draw_translator_ui(frame, sentence, current_stable_sign,
                           stable_frame_count, in_cooldown, remaining_pause,
                           PAUSE_BETWEEN_GESTURES, HOLD_FRAMES_TO_COMMIT)

        cv2.imshow("Sign Language Translator", frame)

        key = cv2.waitKey(1) & 0xFF
        if   key in (ord('q'), ord('Q')):
            break
        elif key == 32 and not in_cooldown and currently_seen_sign:   # SPACE
            sentence.append(currently_seen_sign)
            last_committed_time = time.time()
            speak(currently_seen_sign)
            stable_frame_count  = 0
            current_stable_sign = None
        elif key in (ord('c'), ord('C')):
            sentence.clear(); stable_frame_count = 0; current_stable_sign = None
        elif key in (ord('b'), ord('B'), 8) and sentence:
            sentence.pop()
        elif key in (ord('s'), ord('S')) and sentence:
            speak(" ".join(sentence))

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
