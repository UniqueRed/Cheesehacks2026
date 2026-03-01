"""
cleaner.py — Fast, zero-dependency sign stream cleaner.

Pipeline:
  1. Consecutive duplicate removal   ("the the" → "the")
  2. Near-duplicate window removal   ("we went we" → "we went")
  3. Trailing noise truncation       ("store the the is" → "store")
     — detects when the sentence ends at the last "content" word
     — function words (determiners, prepositions, conjunctions) at the
       tail with no following content word are considered noise
  4. Buffer + flush on silence       (accumulates words, cleans on pause)

No ML, no downloads, no API. Runs in <1ms.
"""

import time
import threading


# ─── LINGUISTIC CONSTANTS ────────────────────────────────────────────────────

# Words exempt from near-duplicate removal — they legitimately
# repeat often in natural sentences ("the store is the best").
DEDUP_EXEMPT = {"the", "a", "an", "is", "are", "was", "were", "to"}

# Words that cannot end a meaningful sentence fragment.
# If the buffer ends with one of these, it's trailing noise.
DANGLING_WORDS = {
    # Determiners
    "the", "a", "an", "this", "that", "these", "those",
    "my", "your", "his", "her", "its", "our", "their",
    # Prepositions
    "to", "of", "in", "on", "at", "by", "for", "with",
    "from", "into", "onto", "upon", "about", "over",
    "under", "through", "between", "among", "around",
    # Conjunctions
    "and", "or", "but", "so", "yet", "nor",
    "because", "although", "if", "when", "while",
    # Auxiliaries that need a following verb
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "must",
    "can", "shall",
    # Other danglers
    "not", "also", "just", "very", "really", "quite",
}

# Duplicate window: how many recent words to check for near-repeats
DEDUP_WINDOW = 4

# Flush after this many seconds of silence
FLUSH_AFTER = 2.0


# ─── CORE CLEANING FUNCTIONS ─────────────────────────────────────────────────

def remove_consecutive_duplicates(words):
    """
    ['we', 'we', 'went'] → ['we', 'went']
    Comparison is case-insensitive; original casing is preserved.
    """
    if not words:
        return []
    result = [words[0]]
    for w in words[1:]:
        if w.lower() != result[-1].lower():
            result.append(w)
    return result


def remove_near_duplicates(words, window=DEDUP_WINDOW):
    """
    Remove a word if it appeared in the last `window` words.
    ['we', 'went', 'we'] → ['we', 'went']
    but preserves intentional repetition far apart.
    Exempt words ('the', 'a', etc.) are never removed by this rule
    since they legitimately repeat in natural sentences.
    """
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
    """
    Remove trailing function words that cannot end a sentence.
    ['we', 'went', 'to', 'the', 'store', 'the', 'the', 'is']
    → ['we', 'went', 'to', 'the', 'store']

    Walks backward from the end, removing dangling words until it
    hits a content word (noun, verb, adjective, adverb, proper noun).
    """
    if not words:
        return []

    result = list(words)
    while result and result[-1].lower() in DANGLING_WORDS:
        result.pop()

    return result


def clean(raw_words):
    """
    Full cleaning pipeline. Takes a list of raw recognized words,
    returns a cleaned string.
    """
    if not raw_words:
        return ""

    # Strip whitespace but preserve original casing for output
    words = [w.strip() for w in raw_words if w.strip()]
    words = remove_consecutive_duplicates(words)
    words = remove_near_duplicates(words)
    words = truncate_trailing_noise(words)

    return " ".join(words)


# ─── STREAM BUFFER ───────────────────────────────────────────────────────────

class StreamCleaner:
    """
    Accumulates raw recognized words in real time.

    Behavior:
      - Every new word is immediately cleaned against the running buffer
        and returned so captions can update live.
      - After SENTENCE_PAUSE seconds of silence, the full accumulated
        sentence is cleaned, spoken as one utterance, then the buffer resets.
      - force_flush() triggers this immediately (e.g. presenter stops).

    This means captions grow word by word in real time, and speech fires
    once per natural sentence pause — not fragmented per chunk.
    """

    def __init__(self, on_word, on_sentence, sentence_pause=4.0):
        """
        on_word(word):      called immediately when a clean word should
                            be appended to live captions.
        on_sentence(text):  called after a pause with the full cleaned
                            sentence for TTS speech.
        sentence_pause:     seconds of silence before speaking the sentence.
        """
        self.on_word      = on_word
        self.on_sentence  = on_sentence
        self.pause        = sentence_pause
        self._raw         = []   # all raw words this sentence
        self._lock        = threading.Lock()
        self._timer       = None

    def push(self, word):
        """
        Push a new raw recognized word.
        Returns the cleaned word to show in captions immediately (or None if noise).
        """
        word = word.strip()
        if not word:
            return

        with self._lock:
            # Check if this word is noise relative to recent raw words
            recent = self._raw[-DEDUP_WINDOW:] if self._raw else []

            # Consecutive duplicate
            if recent and recent[-1].lower() == word.lower():
                self._reset_timer()
                return

            # Near-duplicate (unless exempt)
            if word.lower() not in DEDUP_EXEMPT:
                recent_lower = [w.lower() for w in recent]
                if word.lower() in recent_lower:
                    self._reset_timer()
                    return

            # Word is clean — add to buffer
            self._raw.append(word)

        # Emit immediately for live captions
        self.on_word(word)
        self._reset_timer()

    def force_flush(self):
        """Immediately speak whatever is buffered. Call when presenter stops."""
        self._cancel_timer()
        self._speak_sentence()

    def reset(self):
        """Clear everything silently."""
        self._cancel_timer()
        with self._lock:
            self._raw.clear()

    # ── INTERNALS ────────────────────────────────────────────────────────────

    def _reset_timer(self):
        self._cancel_timer()
        t = threading.Timer(self.pause, self._speak_sentence)
        t.daemon = True
        t.start()
        self._timer = t

    def _cancel_timer(self):
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _speak_sentence(self):
        with self._lock:
            if not self._raw:
                return
            words      = list(self._raw)
            self._raw  = []

        cleaned = clean(words)
        if cleaned:
            print(f"Speaking: '{cleaned}'")
            self.on_sentence(cleaned)