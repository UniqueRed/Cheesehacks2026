import numpy as np


def euclidean_distance(a, b):
    a, b = np.array(a), np.array(b)
    min_len = min(len(a), len(b))
    return np.sqrt(np.sum((a[:min_len] - b[:min_len]) ** 2))


def dtw_distance(seq1, seq2):
    """
    Compute DTW distance between two sequences of landmark frames.
    Each frame is a flattened list of landmark coordinates.
    """
    n, m = len(seq1), len(seq2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = euclidean_distance(seq1[i - 1], seq2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1]
            )

    return dtw_matrix[n, m] / (n + m)


def landmarks_to_vector(landmarks, other_hand=None):
    """
    Convert MediaPipe landmarks to a normalized flat vector.
    Always returns 63 values for single-hand shape.
    If other_hand provided, appends 3 relative wrist values = 66 total.
    Comparison always truncates to min length so mixed cases work fine.
    """
    if not landmarks:
        return None

    points = [(lm.x, lm.y, lm.z) for lm in landmarks]

    # Normalize relative to wrist (landmark 0)
    wrist = points[0]
    normalized = [(p[0] - wrist[0], p[1] - wrist[1], p[2] - wrist[2]) for p in points]

    # Scale by hand size (wrist to middle finger MCP = landmark 9)
    scale = euclidean_distance(
        [normalized[0][0], normalized[0][1], normalized[0][2]],
        [normalized[9][0], normalized[9][1], normalized[9][2]]
    )
    if scale == 0:
        scale = 1
    scaled = [(p[0] / scale, p[1] / scale, p[2] / scale) for p in normalized]

    # Always 63 values (21 landmarks × 3 coords)
    vector = [coord for point in scaled for coord in point]

    # Optionally append relative wrist position (3 more values = 66 total)
    # euclidean_distance handles mismatched lengths so this is safe
    if other_hand:
        other_wrist = (other_hand[0].x, other_hand[0].y, other_hand[0].z)
        rel = (
            (other_wrist[0] - wrist[0]) / scale,
            (other_wrist[1] - wrist[1]) / scale,
            (other_wrist[2] - wrist[2]) / scale,
        )
        vector.extend(rel)

    return vector