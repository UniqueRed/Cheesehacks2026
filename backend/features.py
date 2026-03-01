"""
Extract robust features from hand landmarks.

Instead of raw xyz (which changes with hand position/rotation/distance),
we extract:
- Angles between finger joints (rotation invariant)
- Normalized distances between key landmark pairs (scale invariant)
- Finger extension ratios (bent vs straight)

These features are much more stable and discriminative for gesture recognition.
"""

import numpy as np
import math


def angle_between(v1, v2):
    """Angle in radians between two 3D vectors."""
    v1 = np.array(v1)
    v2 = np.array(v2)
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return math.acos(np.clip(cos, -1.0, 1.0))


def extract_features(landmarks, other_hand=None):
    """
    Extract a robust, invariant feature vector from hand landmarks.
    landmarks: list of landmark objects with .x, .y, .z
    Returns a flat numpy array.
    """
    if not landmarks or len(landmarks) < 21:
        return None

    pts = np.array([(lm.x, lm.y, lm.z) for lm in landmarks])

    # ── 1. Normalize: center on wrist, scale by palm size ──────────────
    wrist = pts[0]
    pts = pts - wrist

    # Palm size = avg distance from wrist to each MCP joint (2,5,9,13,17)
    mcp_joints = [2, 5, 9, 13, 17]
    palm_size = np.mean([np.linalg.norm(pts[j]) for j in mcp_joints])
    if palm_size < 1e-6:
        return None
    pts = pts / palm_size

    features = []

    # ── 2. Finger joint angles (15 angles — 3 per finger) ──────────────
    # Each finger: MCP, PIP, DIP, TIP
    finger_joints = [
        [1, 2, 3, 4],    # thumb
        [5, 6, 7, 8],    # index
        [9, 10, 11, 12], # middle
        [13, 14, 15, 16],# ring
        [17, 18, 19, 20] # pinky
    ]

    for finger in finger_joints:
        for i in range(len(finger) - 2):
            a = pts[finger[i]]
            b = pts[finger[i+1]]
            c = pts[finger[i+2]]
            v1 = a - b
            v2 = c - b
            features.append(angle_between(v1, v2))

    # ── 3. Finger extension ratios (is each finger curled or straight?) ─
    # Ratio of tip-to-wrist distance vs tip-to-MCP distance
    for finger in finger_joints:
        tip = pts[finger[-1]]
        mcp = pts[finger[1]]
        tip_dist = np.linalg.norm(tip)
        mcp_dist = np.linalg.norm(mcp)
        features.append(tip_dist / (mcp_dist + 1e-8))

    # ── 4. Key landmark pair distances (normalized by palm size) ────────
    # Thumb tip to each fingertip (important for many signs)
    thumb_tip = pts[4]
    for tip_idx in [8, 12, 16, 20]:
        features.append(np.linalg.norm(thumb_tip - pts[tip_idx]))

    # Fingertip to fingertip distances
    tips = [8, 12, 16, 20]
    for i in range(len(tips)):
        for j in range(i+1, len(tips)):
            features.append(np.linalg.norm(pts[tips[i]] - pts[tips[j]]))

    # ── 5. Fingertip heights (y relative to palm) ───────────────────────
    palm_normal_y = np.mean(pts[mcp_joints, 1])
    for tip_idx in [4, 8, 12, 16, 20]:
        features.append(pts[tip_idx][1] - palm_normal_y)

    # ── 6. Palm orientation (normal vector components) ───────────────────
    # Cross product of two palm vectors gives orientation
    v1 = pts[5] - pts[0]   # wrist to index MCP
    v2 = pts[17] - pts[0]  # wrist to pinky MCP
    normal = np.cross(v1, v2)
    norm_mag = np.linalg.norm(normal)
    if norm_mag > 1e-6:
        normal = normal / norm_mag
    features.extend(normal.tolist())

    # ── 7. Two-hand relative features ────────────────────────────────────
    if other_hand and len(other_hand) >= 21:
        other_pts = np.array([(lm.x, lm.y, lm.z) for lm in other_hand])
        other_wrist = other_pts[0]

        # Relative wrist position (normalized by palm size of primary hand)
        rel_wrist = (other_wrist - (wrist + wrist)) / palm_size  # already centered
        rel_wrist = (other_wrist - landmarks[0].x) / palm_size
        # Simple: just the raw offset between wrists
        wrist_offset = other_wrist - np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
        wrist_offset = wrist_offset / palm_size
        features.extend(wrist_offset.tolist())

        # Distance between wrists
        features.append(np.linalg.norm(wrist_offset))
    else:
        # Pad with zeros for consistency
        features.extend([0.0, 0.0, 0.0, 0.0])

    return np.array(features, dtype=np.float32)