import numpy as np

def extract_and_normalize_landmarks(landmarks):
    """
    Extracts 21 3D hand landmarks from MediaPipe and normalizes them:
    1. Translates all landmarks relative to the wrist (landmark 0).
    2. Scales by the maximum Euclidean distance from the wrist to achieve scale invariance.
    
    Returns:
        np.ndarray: 1D array of 63 normalized coordinates [x0, y0, z0, x1, y1, z1, ...]
    """
    coords = []
    for lm in landmarks.landmark:
        coords.append([lm.x, lm.y, lm.z])
    coords = np.array(coords)  # Shape: (21, 3)

    # Landmark 0 is the wrist
    wrist = coords[0]
    
    # Translation invariance: Shift origin to wrist
    relative_coords = coords - wrist  # Shape: (21, 3)

    # Scale invariance: compute max distance from wrist
    distances = np.linalg.norm(relative_coords, axis=1)
    max_dist = np.max(distances)

    if max_dist > 1e-6:
        normalized_coords = relative_coords / max_dist
    else:
        normalized_coords = relative_coords

    return normalized_coords.flatten()  # 63 features
