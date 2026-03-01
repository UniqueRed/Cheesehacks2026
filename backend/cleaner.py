"""
cleaner.py — Fast, zero-dependency sign stream cleaner.

Pipeline:
  1. Consecutive duplicate removal   ("the the" → "the")
  2. Near-duplicate window removal   ("we went we" → "we went")
  3. Trailing noise truncation       ("store the the is" → "store")

Flushing:
  Sentences are flushed ONLY via force_flush() — called when the user
  presses E on the frontend. No automatic pause timer. This allows
  continuous signing with keyboard-dictated sentence boundaries.
  The cleaner acts as autocorrect on each flushed chunk.
"""

import threading
from collections import Counter


# ─── LINGUISTIC CONSTANTS ────────────────────────────────────────────────────

DEDUP_EXEMPT = {"the", "a", "an", "is", "are", "was", "were", "to"}

DANGLING_WORDS = {
    "the", "a", "an", "this", "that", "these", "those",
    "my", "your", "his", "her", "its", "our", "their",
    "to", "of", "in", "on", "at", "by", "for", "with",
    "from", "into", "onto", "upon", "about", "over",
    "under", "through", "between", "among", "around",
    "and", "or", "but", "so", "yet", "nor",
    "because", "although", "if", "when", "while",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "must",
    "can", "shall",
    "not", "also", "just", "very", "really", "quite",
}

DEDUP_WINDOW = 4


# ─── CORE CLEANING FUNCTIONS ─────────────────────────────────────────────────

def remove_consecutive_duplicates(words):
    if not words:
        return []
    result = [words[0]]
    for w in words[1:]:
        if w.lower() != result[-1].lower():
            result.append(w)
    return result


def remove_near_duplicates(words, window=DEDUP_WINDOW):
    result = []
    for w in words:
        wl = w.lower()
        if wl in DEDUP_EXEMPT:
            result.append(w)
            continue
        recent_lower = [r.lower() for r in (result[-window:] if len(result) >= window else result)]
        if wl not in recent_lower:
            result.append(w)
    return result


def truncate_trailing_noise(words):
    if not words:
        return []
    result = list(words)
    while result and result[-1].lower() in DANGLING_WORDS:
        result.pop()
    return result


def clean(raw_words):
    if not raw_words:
        return ""
    words = [w.strip() for w in raw_words if w.strip()]
    words = remove_consecutive_duplicates(words)
    words = remove_near_duplicates(words)
    words = truncate_trailing_noise(words)
    return " ".join(words)


# ─── STREAM BUFFER ───────────────────────────────────────────────────────────

class StreamCleaner:
    """
    Accumulates raw recognized words continuously.
    Words are emitted live for captions via on_word.
    Sentences are only flushed via force_flush() (E key press).
    Cleaning runs on flush as autocorrect for that chunk.
    """

    def __init__(self, on_word, on_sentence, sentence_pause=None):
        """
        on_word(word):           called immediately for live captions.
        on_sentence(text, emotion): called on force_flush with cleaned sentence.
        sentence_pause:          ignored — kept for API compatibility.
        """
        self.on_word     = on_word
        self.on_sentence = on_sentence
        self._raw        = []
        self._emotions   = []
        self._lock       = threading.Lock()

    def push(self, word, emotion=None):
        """Push a recognized word. Dedup runs here to keep captions clean."""
        word = word.strip()
        if not word:
            return

        with self._lock:
            recent = self._raw[-DEDUP_WINDOW:] if self._raw else []

            # Consecutive duplicate
            if recent and recent[-1].lower() == word.lower():
                return

            # Near-duplicate (unless exempt)
            if word.lower() not in DEDUP_EXEMPT:
                if word.lower() in [w.lower() for w in recent]:
                    return

            self._raw.append(word)
            self._emotions.append(emotion or "neutral")
            print(f"[Cleaner] + '{word}' ({emotion or 'neutral'})  buffer={len(self._raw)} words")

        self.on_word(word)

    def force_flush(self):
        """Flush current buffer as a sentence (called on E key press)."""
        self._speak_sentence()

    def reset(self):
        """Clear everything silently (called when presenter stops)."""
        with self._lock:
            self._raw.clear()
            self._emotions.clear()

    def _speak_sentence(self):
        with self._lock:
            if not self._raw:
                return
            words    = list(self._raw)
            emotions = list(self._emotions)
            self._raw.clear()
            self._emotions.clear()

        sentence = clean(words)
        if not sentence:
            return

        emotion_counts = Counter(emotions)
        avg_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else "neutral"
        print(f"[Cleaner] Flush → '{sentence}' | emotion={avg_emotion}")
        self.on_sentence(sentence, avg_emotion)