"""
features.py — Finger state encoding for robust gesture recognition.

Instead of raw xyz coordinates (which vary with distance, position, rotation),
we encode each hand frame as a compact, normalized descriptor:

STATIC FEATURES (for static sign matching):
  - 5 finger extension ratios (0=fully curled, 1=fully extended)
  - 5 finger curl angles at PIP joint
  - Thumb-to-fingertip distances (normalized)
  - Palm facing direction (4 components of normal vector)
  - Inter-fingertip spread

DYNAMIC FEATURES (per-frame, for sequence DTW):
  - Same as above but lighter — optimized for temporal comparison

The key insight: all features are normalized so they're invariant to
hand size, distance from camera, and absolute position in frame.
"""

import numpy as np
import math


def _vec(a, b):
    return b - a

def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v

def _angle(a, b, c):
    """Angle at vertex b, formed by points a-b-c."""
    v1 = _norm(a - b)
    v2 = _norm(c - b)
    cos = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return math.acos(cos)

def _pts(landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)


# Landmark index reference (MediaPipe):
# 0=wrist, 1=thumb_cmc, 2=thumb_mcp, 3=thumb_ip, 4=thumb_tip
# 5=index_mcp, 6=index_pip, 7=index_dip, 8=index_tip
# 9=mid_mcp, 10=mid_pip, 11=mid_dip, 12=mid_tip
# 13=ring_mcp, 14=ring_pip, 15=ring_dip, 16=ring_tip
# 17=pinky_mcp, 18=pinky_pip, 19=pinky_dip, 20=pinky_tip

FINGERS = [
    (2, 3, 4),    # thumb:  mcp, ip, tip
    (5, 6, 8),    # index:  mcp, pip, tip
    (9, 10, 12),  # middle: mcp, pip, tip
    (13, 14, 16), # ring:   mcp, pip, tip
    (17, 18, 20), # pinky:  mcp, pip, tip
]

TIPS    = [4, 8, 12, 16, 20]
MCPS    = [2, 5, 9, 13, 17]
PIPS    = [3, 6, 10, 14, 18]


def extract_features(landmarks, other_hand=None):
    """
    Full static feature vector for sign matching.
    Returns a 1D numpy float32 array.
    """
    if not landmarks or len(landmarks) < 21:
        return None

    pts = _pts(landmarks)
    wrist = pts[0].copy()
    pts -= wrist

    # Scale by palm size (wrist to middle MCP)
    palm_size = np.linalg.norm(pts[9])
    if palm_size < 1e-6:
        return None
    pts /= palm_size

    feats = []

    # ── 1. Finger extension ratios ──────────────────────────────────────
    # Tip distance from wrist vs MCP distance from wrist
    # 1.0 = fully extended, ~0.3 = fully curled
    for mcp_i, pip_i, tip_i in FINGERS:
        tip_dist = np.linalg.norm(pts[tip_i])
        mcp_dist = np.linalg.norm(pts[mcp_i])
        feats.append(tip_dist / (mcp_dist + 1e-6))

    # ── 2. PIP joint angles (how bent is each finger?) ──────────────────
    for mcp_i, pip_i, tip_i in FINGERS:
        angle = _angle(pts[mcp_i], pts[pip_i], pts[tip_i])
        feats.append(angle / math.pi)  # normalize to [0, 1]

    # ── 3. Thumb tip to each fingertip distance ─────────────────────────
    thumb_tip = pts[4]
    for tip_i in [8, 12, 16, 20]:
        feats.append(np.linalg.norm(thumb_tip - pts[tip_i]))

    # ── 4. Adjacent fingertip distances (spread / closeness) ────────────
    for i in range(len(TIPS) - 1):
        feats.append(np.linalg.norm(pts[TIPS[i]] - pts[TIPS[i+1]]))

    # ── 5. Palm normal vector (orientation) ─────────────────────────────
    v1 = pts[5]  - pts[0]
    v2 = pts[17] - pts[0]
    normal = np.cross(v1, v2)
    n_mag = np.linalg.norm(normal)
    if n_mag > 1e-6:
        normal /= n_mag
    feats.extend(normal.tolist())

    # ── 6. Fingertip heights relative to palm center ────────────────────
    palm_center_y = np.mean(pts[MCPS, 1])
    for tip_i in TIPS:
        feats.append(pts[tip_i][1] - palm_center_y)

    # ── 7. MCP spread (how wide is the hand open?) ──────────────────────
    mcp_pts = pts[MCPS]
    feats.append(np.linalg.norm(mcp_pts[0] - mcp_pts[-1]))  # thumb MCP to pinky MCP

    # ── 8. Two-hand relative features ───────────────────────────────────
    if other_hand and len(other_hand) >= 21:
        other_pts = _pts(other_hand)
        other_wrist = other_pts[0]
        # Relative wrist position (normalized by primary palm size)
        rel_wrist = (other_wrist - (wrist + wrist)) / palm_size
        rel_wrist = other_wrist - np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
        rel_wrist = rel_wrist / palm_size
        feats.extend(rel_wrist.tolist())
        feats.append(np.linalg.norm(rel_wrist))
    else:
        feats.extend([0.0, 0.0, 0.0, 0.0])

    return np.array(feats, dtype=np.float32)


def extract_dynamic_frame(landmarks, other_hand=None):
    """
    Lightweight per-frame feature vector for dynamic sign DTW.
    Focuses on finger states and relative positions — fast to compute,
    robust for temporal comparison.
    """
    if not landmarks or len(landmarks) < 21:
        return None

    pts = _pts(landmarks)
    wrist = pts[0].copy()
    pts -= wrist

    palm_size = np.linalg.norm(pts[9])
    if palm_size < 1e-6:
        return None
    pts /= palm_size

    feats = []

    # Finger extension ratios (5)
    for mcp_i, pip_i, tip_i in FINGERS:
        tip_dist = np.linalg.norm(pts[tip_i])
        mcp_dist = np.linalg.norm(pts[mcp_i])
        feats.append(tip_dist / (mcp_dist + 1e-6))

    # PIP bend angles (5)
    for mcp_i, pip_i, tip_i in FINGERS:
        feats.append(_angle(pts[mcp_i], pts[pip_i], pts[tip_i]) / math.pi)

    # Palm normal (3) — captures rotation/orientation change over time
    v1 = pts[5]  - pts[0]
    v2 = pts[17] - pts[0]
    normal = np.cross(v1, v2)
    n_mag = np.linalg.norm(normal)
    if n_mag > 1e-6:
        normal /= n_mag
    feats.extend(normal.tolist())

    # Thumb to index tip (1) — key for open/close detection
    feats.append(np.linalg.norm(pts[4] - pts[8]))

    return np.array(feats, dtype=np.float32)


def fingertip_velocity(prev_lms, curr_lms):
    """
    Average velocity of the 5 fingertips between two frames.
    Used for motion onset/offset detection.
    Both args are lists of landmark objects.
    """
    tips = [4, 8, 12, 16, 20]
    total = 0.0
    for i in tips:
        dx = curr_lms[i].x - prev_lms[i].x
        dy = curr_lms[i].y - prev_lms[i].y
        total += (dx**2 + dy**2) ** 0.5
    return total / 5.0