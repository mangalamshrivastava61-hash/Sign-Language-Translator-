# app.py - Flask wrapper around the sign-language pipeline
#
# Serves the web frontend from the `frontend/` folder and exposes
# endpoints that the JavaScript calls:
#
#   GET  /                -> index.html
#   GET  /video_feed      -> MJPEG stream (camera + MediaPipe overlay)
#   GET  /status          -> JSON {sign, confidence, sentence, state, classes}
#   POST /add_sign        -> add current detected sign to sentence
#   POST /clear           -> clear the sentence
#   POST /train           -> retrain the model from collected data
#   POST /delete_last     -> remove last word from sentence
# ---------------------------------------------------------------

import core.bootstrap  # venv auto-redirect, DLL loading

import os, time, pickle, threading
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, Response, jsonify, request, send_from_directory

from core.utils import extract_two_hands_features, normalize_single_hand
from core.config import (
    CONFIDENCE_THRESHOLD,
    PAUSE_BETWEEN_GESTURES,
    HOLD_FRAMES_TO_COMMIT,
    DATA_FILE,
    MODEL_FILE,
    MP_MAX_HANDS,
    MP_DETECTION_CONFIDENCE,
    MP_TRACKING_CONFIDENCE,
)

# ---------------------------------------------------------------
# Flask app - static files served from frontend/
# ---------------------------------------------------------------
app = Flask(__name__, static_folder="frontend", static_url_path="/static")

# ---------------------------------------------------------------
# Load dataset and model (same files as the CLI scripts)
# ---------------------------------------------------------------
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "rb") as f:
        dataset = pickle.load(f)
else:
    dataset = {"features": [], "labels": [], "classes": []}

if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, "rb") as f:
        payload = pickle.load(f)
        clf = payload["model"]
        classes = payload["classes"]
else:
    clf = None
    classes = []

# ---------------------------------------------------------------
# Shared state (protected by a lock for thread safety)
# ---------------------------------------------------------------
lock = threading.Lock()
state = {
    "sign": None,
    "confidence": 0.0,
    "sentence": [],
    "status": "Ready",
    "currently_seen_sign": None,
    "current_stable_sign": None,
    "stable_frame_count": 0,
    "last_committed_time": 0.0,
}

# ---------------------------------------------------------------
# Camera + MediaPipe
# ---------------------------------------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

EXPECTED_FEATURES = getattr(clf, "n_features_in_", 63) if clf else 63

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MP_MAX_HANDS,
    min_detection_confidence=MP_DETECTION_CONFIDENCE,
    min_tracking_confidence=MP_TRACKING_CONFIDENCE,
)

cap = None
cap_lock = threading.Lock()


def get_camera():
    global cap
    with cap_lock:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(0)
        return cap


def generate_frames():
    """Yield MJPEG frames with MediaPipe overlay."""
    camera = get_camera()

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        now = time.time()

        with lock:
            time_since = now - state["last_committed_time"]
            in_cooldown = time_since < PAUSE_BETWEEN_GESTURES
            remaining = max(0.0, PAUSE_BETWEEN_GESTURES - time_since)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        detected_sign = None
        confidence = 0.0
        num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

        if num_hands > 0:
            all_x, all_y = [], []
            for idx, hand_lm in enumerate(results.multi_hand_landmarks):
                color = (0, 140, 255) if in_cooldown else ((0, 255, 255) if idx == 0 else (255, 0, 255))
                mp_drawing.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=color, thickness=2),
                )
                for lm in hand_lm.landmark:
                    all_x.append(int(lm.x * w))
                    all_y.append(int(lm.y * h))

            if not in_cooldown and clf is not None:
                if EXPECTED_FEATURES == 126:
                    features = extract_two_hands_features(results)
                else:
                    features = normalize_single_hand(results.multi_hand_landmarks[0])

                features = np.expand_dims(features, axis=0)
                probs = clf.predict_proba(features)[0]
                best_idx = np.argmax(probs)
                confidence = float(probs[best_idx])
                candidate = clf.classes_[best_idx]

                if confidence >= CONFIDENCE_THRESHOLD:
                    detected_sign = candidate

            # Bounding box
            if all_x and all_y:
                x1 = max(0, min(all_x) - 25)
                x2 = min(w, max(all_x) + 25)
                y1 = max(0, min(all_y) - 25)
                y2 = min(h, max(all_y) + 25)

                if in_cooldown:
                    box_color = (0, 140, 255)
                    tag = "Paused ({:.1f}s)".format(remaining)
                elif detected_sign:
                    box_color = (0, 255, 0)
                    tag = "{} ({:.0f}%)".format(detected_sign, confidence * 100)
                else:
                    box_color = (120, 120, 120)
                    tag = "Analyzing..."

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.rectangle(frame, (x1, max(0, y1 - 32)), (x1 + len(tag) * 12 + 10, y1), box_color, -1)
                cv2.putText(frame, tag, (x1 + 5, y1 - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2)

        # Update shared state
        with lock:
            state["confidence"] = confidence
            if in_cooldown:
                state["sign"] = None
                state["currently_seen_sign"] = None
                state["status"] = "Cooldown ({:.1f}s)".format(remaining)
            elif detected_sign:
                state["sign"] = detected_sign
                state["currently_seen_sign"] = detected_sign
                state["status"] = "Gesture detected"

                if detected_sign == state["current_stable_sign"]:
                    state["stable_frame_count"] += 1
                else:
                    state["current_stable_sign"] = detected_sign
                    state["stable_frame_count"] = 1

                if state["stable_frame_count"] >= HOLD_FRAMES_TO_COMMIT:
                    state["sentence"].append(detected_sign)
                    state["last_committed_time"] = time.time()
                    state["stable_frame_count"] = 0
                    state["current_stable_sign"] = None
                    state["status"] = "Added: {}".format(detected_sign)
            else:
                if num_hands > 0:
                    state["sign"] = None
                    state["status"] = "Analyzing"
                else:
                    state["sign"] = None
                    state["currently_seen_sign"] = None
                    state["status"] = "Ready"
                state["stable_frame_count"] = max(0, state["stable_frame_count"] - 1)
                if state["stable_frame_count"] == 0:
                    state["current_stable_sign"] = None

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


# ---------------------------------------------------------------
# Routes
# ---------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    with lock:
        return jsonify({
            "sign": state["sign"],
            "confidence": state["confidence"],
            "sentence": " ".join(state["sentence"]) if state["sentence"] else "",
            "word_count": len(state["sentence"]),
            "state": state["status"],
            "classes": list(classes) if len(classes) else [],
        })


@app.route("/add_sign", methods=["POST"])
def add_sign():
    with lock:
        sign = state.get("currently_seen_sign")
        now = time.time()
        in_cooldown = (now - state["last_committed_time"]) < PAUSE_BETWEEN_GESTURES
        if sign and not in_cooldown:
            state["sentence"].append(sign)
            state["last_committed_time"] = now
            state["stable_frame_count"] = 0
            state["current_stable_sign"] = None
            return jsonify({"ok": True, "added": sign})
    return jsonify({"ok": False, "reason": "No sign detected or in cooldown"})


@app.route("/clear", methods=["POST"])
def clear():
    with lock:
        state["sentence"].clear()
        state["stable_frame_count"] = 0
        state["current_stable_sign"] = None
    return jsonify({"ok": True})


@app.route("/delete_last", methods=["POST"])
def delete_last():
    with lock:
        if state["sentence"]:
            removed = state["sentence"].pop()
            return jsonify({"ok": True, "removed": removed})
    return jsonify({"ok": False, "reason": "Sentence is empty"})


@app.route("/train", methods=["POST"])
def train():
    global clf, classes

    if not os.path.exists(DATA_FILE):
        return jsonify({"ok": False, "reason": "No dataset found. Run 01_collect_data.py first."}), 400

    try:
        with open(DATA_FILE, "rb") as f:
            ds = pickle.load(f)

        X = np.array(ds["features"])
        y = np.array(ds["labels"])
        unique = np.unique(y)

        if len(unique) < 2:
            return jsonify({"ok": False, "reason": "Need at least 2 different signs to train."}), 400

        from sklearn.ensemble import RandomForestClassifier
        from core.config import N_ESTIMATORS

        new_clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=42, n_jobs=-1)
        new_clf.fit(X, y)

        os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
        with open(MODEL_FILE, "wb") as f:
            pickle.dump({"model": new_clf, "classes": unique, "feature_dim": X.shape[1]}, f)

        with lock:
            clf = new_clf
            classes = unique

        return jsonify({"ok": True, "accuracy": "Model trained", "classes": list(unique)})
    except Exception as e:
        return jsonify({"ok": False, "reason": str(e)}), 500


# ---------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Sign Language Translator - Web Interface")
    print("Model loaded: {}".format(clf is not None))
    if clf:
        print("Classes ({}): {}".format(len(classes), list(classes)))
    print("Open http://localhost:5000 in your browser")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
