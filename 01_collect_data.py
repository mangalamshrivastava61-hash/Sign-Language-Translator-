import core.bootstrap                                 # venv redirect + DLL setup
import os, cv2, pickle
import mediapipe as mp
from core.config import (BASE_DIR, DATA_FILE, SAMPLES_PER_GESTURE,
                         MP_MAX_HANDS, MP_DETECTION_CONFIDENCE, MP_TRACKING_CONFIDENCE)
from core.utils import extract_two_hands_features
from core.ui   import draw_collector_ui

# ── Setup ─────────────────────────────────────────────────────────────────────
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

# Load existing dataset
if os.path.exists(DATA_FILE):
    try:
        with open(DATA_FILE, 'rb') as f:
            ds = pickle.load(f)
        all_features  = ds.get('features', [])
        all_labels    = ds.get('labels',   [])
        saved_classes = list(dict.fromkeys(all_labels))
    except Exception:
        all_features, all_labels, saved_classes = [], [], []
else:
    all_features, all_labels, saved_classes = [], [], []

# MediaPipe
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MP_MAX_HANDS,
    min_detection_confidence=MP_DETECTION_CONFIDENCE,
    min_tracking_confidence=MP_TRACKING_CONFIDENCE,
)

# ── Welcome ───────────────────────────────────────────────────────────────────
print("=" * 65)
print("TWO-HAND SIGN LANGUAGE DATASET RECORDER")
print(f"Current words ({len(saved_classes)}): {saved_classes}" if saved_classes else "Dataset is empty.")
print("=" * 65)

# ── Recording loop ────────────────────────────────────────────────────────────
while True:
    word = input("\nWord to record (or 'done' to finish): ").strip()
    if not word or word.lower() == "done":
        break

    # Handle overwrite
    if word in saved_classes:
        if input(f"'{word}' exists! Overwrite? (y/n, default y): ").strip().lower() != "n":
            all_features  = [f for f, l in zip(all_features, all_labels) if l != word]
            all_labels    = [l for l in all_labels if l != word]
            saved_classes.remove(word)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        continue

    samples, is_capturing = [], False
    print(f"Camera open! Hold SPACEBAR to record '{word}'.")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame    = cv2.flip(frame, 1)
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results  = hands.process(rgb)
            num_det  = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

            # Draw hand skeletons
            if results.multi_hand_landmarks:
                for idx, lm in enumerate(results.multi_hand_landmarks):
                    color = (0, 255, 255) if idx == 0 else (255, 0, 255)
                    mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(0, 200, 0) if is_capturing else color, thickness=2))

            # Capture sample when spacebar held
            if is_capturing:
                h_frame, w_frame = frame.shape[:2]
                if num_det > 0:
                    samples.append((extract_two_hands_features(results), word))
                    cv2.rectangle(frame, (0, 0), (w_frame, h_frame), (0, 255, 0), 6)
                else:
                    cv2.rectangle(frame, (0, 0), (w_frame, h_frame), (0, 0, 255), 6)
                    cv2.putText(frame, "NO HANDS IN VIEW!", (w_frame // 2 - 170, h_frame // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Done collecting?
            if len(samples) >= SAMPLES_PER_GESTURE:
                for feat, lbl in samples:
                    all_features.append(feat)
                    all_labels.append(lbl)
                if word not in saved_classes:
                    saved_classes.append(word)
                print(f"Captured {len(samples)} samples for '{word}'!")
                break

            draw_collector_ui(frame, word, num_det, len(samples), SAMPLES_PER_GESTURE, is_capturing)
            cv2.imshow(f"Record Sign – {word}", frame)

            key = cv2.waitKey(20) & 0xFF
            is_capturing = (key == 32)
            if key in (ord('q'), ord('Q')):
                print("Cancelled.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    with open(DATA_FILE, 'wb') as f:
        pickle.dump({'features': all_features, 'labels': all_labels, 'classes': saved_classes}, f)
    print(f"Dataset saved! Words: {saved_classes}")

hands.close()
print(f"\nDone! {len(all_features)} samples across {len(saved_classes)} words.")
print("Run '02_train_model.py' to train your model.")
