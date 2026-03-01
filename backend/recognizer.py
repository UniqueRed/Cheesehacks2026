import json
import os
import numpy as np
from features import extract_features
from dtw import dtw_distance, landmarks_to_vector

TEMPLATES_FILE = "templates.json"
RECORDINGS_PER_SIGN = 15   # shown in UI as target
MIN_RECORDINGS_TO_TRAIN = 3  # start training after just 3
MIN_CONFIDENCE = 0.55
DYNAMIC_THRESHOLD = 12.0


def augment(vec, n=5, noise=0.03):
    """Generate n slightly perturbed copies of a feature vector."""
    result = [vec]
    for _ in range(n):
        noisy = vec + np.random.normal(0, noise, size=vec.shape)
        result.append(noisy)
    return result


class GestureRecognizer:
    def __init__(self):
        self.templates = {}
        self.static_classifier = None
        self.label_map = {}        # int -> name
        self.reverse_map = {}      # name -> int
        self.load_templates()
        self._rebuild_classifier()

    def load_templates(self):
        if os.path.exists(TEMPLATES_FILE):
            try:
                with open(TEMPLATES_FILE, "r") as f:
                    self.templates = json.load(f)
                print(f"Loaded {len(self.templates)} templates")
            except Exception as e:
                print(f"Load error: {e}")
                self.templates = {}

    def save_templates(self):
        with open(TEMPLATES_FILE, "w") as f:
            json.dump(self.templates, f)

    def _rebuild_classifier(self):
        self.static_classifier = None
        self.label_map = {}
        self.reverse_map = {}

        X, y = [], []
        label_idx = 0
        for name, t in self.templates.items():
            if t["type"] != "static":
                continue
            recs = t.get("recordings", [])
            if len(recs) < MIN_RECORDINGS_TO_TRAIN:
                continue
            self.label_map[label_idx] = name
            self.reverse_map[name] = label_idx
            for rec in recs:
                vec = np.array(rec, dtype=np.float32)
                for aug_vec in augment(vec, n=4, noise=0.02):
                    X.append(aug_vec)
                    y.append(label_idx)
            label_idx += 1

        if len(set(y)) < 2:
            print(f"Not enough classes to train ({len(set(y))} classes)")
            return

        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline

            clf = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    min_samples_split=2,
                    random_state=42,
                    n_jobs=-1,
                ))
            ])
            clf.fit(X, y)
            self.static_classifier = clf
            print(f"Classifier trained: {len(set(y))} classes, {len(X)} samples (with augmentation)")
        except Exception as e:
            print(f"Training error: {e}")

    def add_static_template(self, name, landmarks, other_hand=None):
        vec = extract_features(landmarks, other_hand)
        if vec is None:
            return False, 0

        if name not in self.templates:
            self.templates[name] = {"type": "static", "recordings": []}

        self.templates[name]["recordings"].append(vec.tolist())
        self.save_templates()
        count = len(self.templates[name]["recordings"])
        self._rebuild_classifier()
        return True, count

    def add_dynamic_template(self, name, frame_sequence, other_hand_sequence=None):
        vectors = []
        for i, lm in enumerate(frame_sequence):
            other = other_hand_sequence[i] if other_hand_sequence and i < len(other_hand_sequence) else None
            v = landmarks_to_vector(lm, other)
            if v is not None:
                vectors.append(v)
        if len(vectors) < 3:
            return False, 0

        if name not in self.templates:
            self.templates[name] = {"type": "dynamic", "recordings": []}

        t = self.templates[name]
        t["recordings"].append(vectors)

        if len(t["recordings"]) > 1:
            best_idx, best_avg = 0, float("inf")
            for i, ri in enumerate(t["recordings"]):
                avg = np.mean([dtw_distance(ri, rj) for j, rj in enumerate(t["recordings"]) if i != j])
                if avg < best_avg:
                    best_avg, best_idx = avg, i
            t["centroid"] = t["recordings"][best_idx]
            t["std"] = float(best_avg)
        else:
            t["centroid"] = vectors
            t["std"] = 0

        self.save_templates()
        return True, len(t["recordings"])

    def recognize_static(self, landmarks, other_hand=None):
        if self.static_classifier is None:
            return None, 0

        vec = extract_features(landmarks, other_hand)
        if vec is None:
            return None, 0

        try:
            proba = self.static_classifier.predict_proba([vec])[0]
            best_idx = int(np.argmax(proba))
            confidence = float(proba[best_idx])
            if confidence >= MIN_CONFIDENCE and best_idx in self.label_map:
                return self.label_map[best_idx], confidence
        except Exception as e:
            print(f"Recognition error: {e}")

        return None, 0

    def recognize_dynamic(self, frame_sequence, other_hand_sequence=None):
        vectors = []
        for i, lm in enumerate(frame_sequence):
            other = other_hand_sequence[i] if other_hand_sequence and i < len(other_hand_sequence) else None
            v = landmarks_to_vector(lm, other)
            if v is not None:
                vectors.append(v)

        if len(vectors) < 3:
            return None, 0

        best_match, best_score = None, float("inf")
        for name, t in self.templates.items():
            if t["type"] != "dynamic" or not t.get("centroid"):
                continue
            dist = dtw_distance(vectors, t["centroid"])
            if dist < best_score:
                best_score, best_match = dist, name

        if best_match:
            t = self.templates[best_match]
            threshold = DYNAMIC_THRESHOLD + t.get("std", 0) * 2.0
            if best_score < threshold:
                confidence = max(0, 1 - (best_score / threshold))
                if confidence >= MIN_CONFIDENCE:
                    return best_match, confidence

        return None, 0

    def get_all_templates(self):
        return {
            name: {
                "type": t["type"],
                "recordings": len(t.get("recordings", [])),
                "needed": RECORDINGS_PER_SIGN,
                "ready": len(t.get("recordings", [])) >= MIN_RECORDINGS_TO_TRAIN
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