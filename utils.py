import numpy as np

def normalize_single_hand(landmarks):
    """
    Normalizes 21 3D landmarks of a single hand:
    1. Shifts origin to wrist (landmark 0) for translation invariance.
    2. Scales by maximum distance from wrist for scale invariance.
    Returns:
        np.ndarray: 63-element vector [x0, y0, z0, ..., x20, y20, z20]
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])  # (21, 3)
    wrist = coords[0]
    relative_coords = coords - wrist
    distances = np.linalg.norm(relative_coords, axis=1)
    max_dist = np.max(distances)
    
    if max_dist > 1e-6:
        normalized = relative_coords / max_dist
    else:
        normalized = relative_coords
        
    return normalized.flatten()

def extract_two_hands_features(results):
    """
    Extracts features for up to 2 hands (Left and Right).
    Produces a fixed-size 126-dimensional feature vector:
      - Indices [0:63]   -> Left Hand (63 features, or zeros if absent)
      - Indices [63:126] -> Right Hand (63 features, or zeros if absent)
      
    This allows the model to handle:
      - Single-hand gestures (Left or Right)
      - Two-hand gestures (Both hands together)
    """
    left_hand = np.zeros(63, dtype=np.float32)
    right_hand = np.zeros(63, dtype=np.float32)

    if results.multi_hand_landmarks:
        num_hands = len(results.multi_hand_landmarks)
        for idx in range(num_hands):
            landmarks = results.multi_hand_landmarks[idx]
            norm_coords = normalize_single_hand(landmarks)

            # Determine hand label from MediaPipe classification
            hand_label = "Right"
            if results.multi_handedness and idx < len(results.multi_handedness):
                hand_label = results.multi_handedness[idx].classification[0].label

            if hand_label == "Left" and np.all(left_hand == 0):
                left_hand = norm_coords
            elif hand_label == "Right" and np.all(right_hand == 0):
                right_hand = norm_coords
            else:
                # Fallback if both classified as same hand: put in the empty slot
                if np.all(left_hand == 0):
                    left_hand = norm_coords
                else:
                    right_hand = norm_coords

    return np.concatenate([left_hand, right_hand])  # exactly 126 elements
