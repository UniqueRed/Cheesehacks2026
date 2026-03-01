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
    Accumulates raw recognized words in real time, then cleans and
    flushes them after a silence period or on demand.

    Usage:
        cleaner = StreamCleaner(on_flushed=my_callback)
        cleaner.push("we")
        cleaner.push("we")
        cleaner.push("went")
        # ... 2s of silence ...
        # my_callback("we went") is called automatically

        # Or manually:
        cleaner.force_flush()
    """

    def __init__(self, on_flushed, flush_after=FLUSH_AFTER):
        """
        on_flushed: callable(cleaned_text: str) — called when a chunk is ready.
        flush_after: seconds of silence before auto-flush.
        """
        self.on_flushed  = on_flushed
        self.flush_after = flush_after
        self._buffer     = []
        self._lock       = threading.Lock()
        self._timer      = None

    def push(self, word):
        """
        Push a newly recognized word into the buffer.
        Resets the silence timer.
        """
        word = word.strip()
        if not word:
            return

        with self._lock:
            self._buffer.append(word)

        self._reset_timer()

    def force_flush(self):
        """
        Immediately clean and emit whatever is buffered.
        Call when presenter stops.
        """
        self._cancel_timer()
        self._do_flush()

    def reset(self):
        """Clear buffer without emitting."""
        self._cancel_timer()
        with self._lock:
            self._buffer.clear()

    # ── INTERNALS ────────────────────────────────────────────────────────────

    def _reset_timer(self):
        self._cancel_timer()
        t = threading.Timer(self.flush_after, self._do_flush)
        t.daemon = True
        t.start()
        self._timer = t

    def _cancel_timer(self):
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _do_flush(self):
        with self._lock:
            if not self._buffer:
                return
            words          = list(self._buffer)
            self._buffer   = []

        cleaned = clean(words)
        if cleaned:
            print(f"Cleaner: {words} → '{cleaned}'")
            self.on_flushed(cleaned)


# ─── QUICK TEST ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cases = [
        (["we", "we", "went", "to", "the", "the", "store", "the", "the", "is"],
         "we went to the store"),
        (["hello", "hello", "my", "name", "is", "is", "john"],
         "hello my name is john"),
        (["built", "on", "on", "react"],
         "built on react"),
        (["the", "the", "the"],
         ""),
        (["we", "are", "presenting", "today"],
         "we are presenting today"),
        (["this", "is", "a", "great", "project", "a", "a", "the"],
         "this is a great project"),
    ]

    print("Running cleaner tests...\n")
    all_passed = True
    for words, expected in cases:
        result = clean(words)
        status = "✓" if result == expected else "✗"
        if result != expected:
            all_passed = False
        print(f"  {status} {words}")
        print(f"      got:      '{result}'")
        if result != expected:
            print(f"      expected: '{expected}'")
        print()

    print("All tests passed!" if all_passed else "Some tests failed.")