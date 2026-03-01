"""
recognizer.py

Static signs:  SVM with RBF kernel on normalized finger-state features.
               SVM is dramatically better than Random Forest for this problem:
               - Works well with small datasets (5-15 examples per class)
               - The RBF kernel finds non-linear boundaries in feature space
               - Probability calibration gives reliable confidence scores
               - Much less prone to overfitting with few samples

Dynamic signs: DTW on per-frame finger-state vectors (not raw coordinates).
               Comparing finger-state sequences is far more robust than comparing
               raw xyz because it's invariant to hand position/size/distance.
               Centroid is the medoid (most central real example) not a mean,
               which avoids artifacts from averaging sequences of different lengths.
"""

import json
import os
import numpy as np
from features import extract_features, extract_dynamic_frame

TEMPLATES_FILE       = "templates.json"
RECORDINGS_PER_SIGN  = 15
MIN_RECORDINGS       = 3    # minimum to start classifying
MIN_CONFIDENCE       = 0.60


# ─── DTW ─────────────────────────────────────────────────────────────────────

def _dtw(seq1, seq2):
    """
    DTW distance between two sequences of feature vectors.
    No band constraint — sequences from the same sign can vary significantly
    in length (fast vs slow execution), and a tight band causes inf distances.
    Normalized by (n + m) so longer sequences don't dominate.
    """
    n, m = len(seq1), len(seq2)
    if n == 0 or m == 0:
        return float("inf")

    INF = float("inf")
    dtw = np.full((n + 1, m + 1), INF)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = float(np.linalg.norm(
                np.array(seq1[i-1], dtype=np.float32) -
                np.array(seq2[j-1], dtype=np.float32)
            ))
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])

    return float(dtw[n, m]) / (n + m)


def _augment_static(vec, n=6, noise=0.015):
    """Generate slightly perturbed copies of a static feature vector."""
    result = [vec]
    for _ in range(n):
        result.append(vec + np.random.normal(0, noise, size=vec.shape))
    return result


def _augment_dynamic(seq, n=3, noise=0.01):
    """Generate slightly perturbed copies of a dynamic sequence."""
    result = [seq]
    for _ in range(n):
        noisy = [v + np.random.normal(0, noise, size=v.shape) for v in seq]
        result.append(noisy)
    return result


# ─── MAIN CLASS ──────────────────────────────────────────────────────────────

class GestureRecognizer:
    def __init__(self):
        self.templates      = {}
        self.classifier     = None   # SVM pipeline for static signs
        self.label_map      = {}     # int -> name
        self.reverse_map    = {}     # name -> int
        self.load_templates()
        self._rebuild_classifier()

    # ── PERSISTENCE ──────────────────────────────────────────────────────────

    def load_templates(self):
        if os.path.exists(TEMPLATES_FILE):
            try:
                with open(TEMPLATES_FILE) as f:
                    self.templates = json.load(f)
                print(f"Loaded {len(self.templates)} sign templates")
            except Exception as e:
                print(f"Load error: {e}")
                self.templates = {}

    def save_templates(self):
        def _sanitize(obj):
            """Recursively replace nan/inf with safe fallback values."""
            if isinstance(obj, float):
                import math
                if math.isnan(obj) or math.isinf(obj):
                    return 0.0
                return obj
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_sanitize(v) for v in obj]
            return obj

        safe = _sanitize(self.templates)
        with open(TEMPLATES_FILE, "w") as f:
            json.dump(safe, f)

    # ── CLASSIFIER ───────────────────────────────────────────────────────────

    def _rebuild_classifier(self):
        self.classifier  = None
        self.label_map   = {}
        self.reverse_map = {}

        X, y     = [], []
        label_idx = 0

        for name, t in self.templates.items():
            if t["type"] != "static":
                continue
            recs = t.get("recordings", [])
            if len(recs) < MIN_RECORDINGS:
                continue

            self.label_map[label_idx]  = name
            self.reverse_map[name]     = label_idx

            for rec in recs:
                vec = np.array(rec, dtype=np.float32)
                for aug in _augment_static(vec, n=5, noise=0.012):
                    X.append(aug)
                    y.append(label_idx)

            label_idx += 1

        n_classes = len(set(y))
        if n_classes == 0:
            print("No static signs ready to train yet")
            return

        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline

            if n_classes == 1:
                # Single class — KNN with distance weighting.
                # Will recognize that one sign if it's close enough to training examples.
                from sklearn.neighbors import KNeighborsClassifier
                clf = Pipeline([
                    ("scaler", StandardScaler()),
                    ("knn",    KNeighborsClassifier(
                        n_neighbors=min(5, len(X)),
                        weights="distance",
                        metric="euclidean")),
                ])
                clf.fit(X, y)
                self.classifier = clf
                print(f"KNN trained (1 class): {len(X)} samples")
            else:
                from sklearn.svm import SVC
                svm = SVC(
                    kernel="rbf",
                    C=10.0,
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                    random_state=42,
                )
                clf = Pipeline([
                    ("scaler", StandardScaler()),
                    ("svm",    svm),
                ])
                clf.fit(X, y)
                self.classifier = clf
                print(f"SVM trained: {n_classes} classes, {len(X)} samples")
        except Exception as e:
            print(f"Training error: {e}")
            import traceback; traceback.print_exc()

    # ── STATIC RECORDING ─────────────────────────────────────────────────────

    def add_static_template(self, name, landmarks, other_hand=None):
        vec = extract_features(landmarks, other_hand)
        if vec is None:
            return False, 0

        if name not in self.templates:
            self.templates[name] = {"type": "static", "recordings": []}
        elif self.templates[name]["type"] != "static":
            return False, 0

        self.templates[name]["recordings"].append(vec.tolist())
        count = len(self.templates[name]["recordings"])
        self.save_templates()
        self._rebuild_classifier()
        return True, count

    # ── STATIC RECOGNITION ───────────────────────────────────────────────────

    def recognize_static(self, landmarks, other_hand=None):
        if self.classifier is None:
            return None, 0.0

        vec = extract_features(landmarks, other_hand)
        if vec is None:
            return None, 0.0

        try:
            n_classes = len(self.label_map)

            if n_classes == 1:
                # Single class: use distance to nearest neighbor as confidence proxy.
                # Transform the vector, compute distance to training set.
                scaler    = self.classifier.named_steps["scaler"]
                knn       = self.classifier.named_steps["knn"]
                vec_scaled = scaler.transform([vec])
                dists, _   = knn.kneighbors(vec_scaled, n_neighbors=1)
                dist       = float(dists[0][0])
                # Convert distance to confidence: close = high confidence
                conf = float(np.exp(-dist * 0.5))
                if conf >= MIN_CONFIDENCE:
                    return self.label_map[0], conf
            else:
                proba    = self.classifier.predict_proba([vec])[0]
                best_idx = int(np.argmax(proba))
                conf     = float(proba[best_idx])
                if conf >= MIN_CONFIDENCE and best_idx in self.label_map:
                    return self.label_map[best_idx], conf

        except Exception as e:
            print(f"Static recognition error: {e}")

        return None, 0.0

    # ── DYNAMIC RECORDING ────────────────────────────────────────────────────

    def add_dynamic_template(self, name, frame_sequence, other_hand_sequence=None):
        """
        frame_sequence: list of landmark objects (one per frame).
        Converts to finger-state vectors and stores.
        """
        vectors = []
        for i, lms in enumerate(frame_sequence):
            other = (other_hand_sequence[i]
                     if other_hand_sequence and i < len(other_hand_sequence)
                     else None)
            v = extract_dynamic_frame(lms, other)
            if v is not None:
                vectors.append(v)

        if len(vectors) < 4:
            return False, 0

        if name not in self.templates:
            self.templates[name] = {"type": "dynamic", "recordings": []}
        elif self.templates[name]["type"] != "dynamic":
            return False, 0

        t = self.templates[name]

        # Store as list of lists (JSON-serializable)
        t["recordings"].append([v.tolist() for v in vectors])

        # Recompute medoid centroid and adaptive threshold
        self._recompute_dynamic_centroid(name)

        self.save_templates()
        return True, len(t["recordings"])

    def _recompute_dynamic_centroid(self, name):
        t    = self.templates[name]
        recs = t["recordings"]
        n    = len(recs)

        # Always set centroid to first recording as baseline
        t["centroid"]  = recs[0]

        if n == 1:
            # Generous fixed threshold — will tighten as more recordings come in
            t["threshold"] = 8.0
            t["std"]       = 0.0
            print(f"Dynamic '{name}': 1 recording, threshold=8.0 (add more for better accuracy)")
            return

        # Convert to numpy sequences for DTW
        seqs = [[np.array(v, dtype=np.float32) for v in r] for r in recs]

        # Compute full pairwise DTW distance matrix
        dmat = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = _dtw(seqs[i], seqs[j])
                dmat[i, j] = d
                dmat[j, i] = d

        # Medoid = recording with lowest total distance to all others
        totals     = dmat.sum(axis=1)
        medoid_idx = int(np.argmin(totals))
        t["centroid"] = recs[medoid_idx]

        # Collect distances from medoid to every other recording, filter any inf
        all_dists   = [float(dmat[medoid_idx, j]) for j in range(n) if j != medoid_idx]
        other_dists = [d for d in all_dists if np.isfinite(d)]

        if not other_dists:
            # All distances were inf — sequences too different, use generous default
            t["threshold"] = 8.0
            t["std"]       = 0.0
            print(f"  Warning: all DTW distances were inf, using default threshold")
            return

        mean_d = float(np.mean(other_dists))
        std_d  = float(np.std(other_dists)) if len(other_dists) > 1 else mean_d * 0.5

        # Threshold: mean + 1.5*std, floored at mean*1.3 and absolute min of 1.0
        t["threshold"] = float(max(mean_d + 1.5 * std_d, mean_d * 1.3, 1.0))
        t["std"]       = std_d

        print(f"Dynamic '{name}': {n} recordings, medoid={medoid_idx}, "
              f"mean={mean_d:.3f}, std={std_d:.3f}, threshold={t['threshold']:.3f}")

    # ── DYNAMIC RECOGNITION ──────────────────────────────────────────────────

    def recognize_dynamic(self, frame_sequence, other_hand_sequence=None):
        """
        frame_sequence: list of landmark objects.
        Returns (name, confidence) or (None, 0).
        """
        vectors = []
        for i, lms in enumerate(frame_sequence):
            other = (other_hand_sequence[i]
                     if other_hand_sequence and i < len(other_hand_sequence)
                     else None)
            v = extract_dynamic_frame(lms, other)
            if v is not None:
                vectors.append(v)

        if len(vectors) < 4:
            return None, 0.0

        best_name  = None
        best_score = float("inf")
        best_thresh = 1.0

        for name, t in self.templates.items():
            if t["type"] != "dynamic" or not t.get("centroid"):
                continue
            if len(t.get("recordings", [])) < MIN_RECORDINGS:
                continue

            centroid = [np.array(v, dtype=np.float32) for v in t["centroid"]]
            dist     = _dtw(vectors, centroid)

            if dist < best_score:
                best_score  = dist
                best_name   = name
                best_thresh = t.get("threshold", 8.0)

        if best_name is None:
            return None, 0.0

        if best_score <= best_thresh:
            # Confidence: 1.0 at dist=0, approaches 0 at dist=threshold
            confidence = max(0.0, 1.0 - (best_score / best_thresh))
            if confidence >= MIN_CONFIDENCE:
                return best_name, confidence

        return None, 0.0

    # ── MANAGEMENT ───────────────────────────────────────────────────────────

    def get_all_templates(self):
        return {
            name: {
                "type":       t["type"],
                "recordings": len(t.get("recordings", [])),
                "needed":     RECORDINGS_PER_SIGN,
                "ready":      len(t.get("recordings", [])) >= MIN_RECORDINGS,
            }
            for name, t in self.templates.items()
        }

    def delete_template(self, name):
        if name in self.templates:
            del self.templates[name]
            self.save_templates()
            self._rebuild_classifier()
            return True
        return False