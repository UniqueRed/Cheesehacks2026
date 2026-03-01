"""
main.py — SignSpeak WebSocket backend.

Architecture:
  Static signs:  Voting window (16 frames, 72% agreement) → SVM classifier.
  Dynamic signs: Motion-segmented capture → DTW on finger-state sequences.
  Cleaning:      StreamCleaner buffers raw words, flushes cleaned chunks
                 after 2s of silence or when presenter stops.
"""

import json
import asyncio
import os
import numpy as np
from collections import deque, Counter
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from recognizer import GestureRecognizer
from features import extract_dynamic_frame, fingertip_velocity
from cleaner import StreamCleaner
from tts_service import TTSService
import base64

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

recognizer = GestureRecognizer()

# Initialize TTS service (will raise error if credentials not set)
# Try to find credentials file automatically

# Check if GOOGLE_APPLICATION_CREDENTIALS is already set
if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    # Look for credentials file in common locations
    possible_paths = [
        Path(__file__).parent.parent / "signspeak-488902-b29067f64881.json",
        Path(__file__).parent / "signspeak-488902-b29067f64881.json",
        Path.cwd() / "signspeak-488902-b29067f64881.json",
    ]
    
    for cred_path in possible_paths:
        if cred_path.exists():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred_path.absolute())
            print(f"[TTS] Found credentials file at: {cred_path.absolute()}")
            break
    else:
        print(f"[TTS] Warning: No credentials file found. Checked paths: {possible_paths}")

try:
    tts_service = TTSService()
    print(f"[TTS] TTS service initialized successfully")
except Exception as e:
    print(f"[TTS] ERROR: TTS service initialization failed: {e}")
    print(f"[TTS] GOOGLE_APPLICATION_CREDENTIALS: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")
    import traceback
    traceback.print_exc()
    tts_service = None

# ─── TUNING ───────────────────────────────────────────────────────────────────
VOTE_WINDOW      = 10    # was 16: fires in ~0.67s at 15fps instead of ~1s
VOTE_THRESHOLD   = 0.60  # was 0.72: 6/10 frames must agree, more lenient
DYN_ONSET_VEL    = 0.015  # was 0.018: detect motion onset sooner
DYN_END_VEL      = 0.012  # was 0.010: slightly easier to register as quiet
DYN_END_FRAMES   = 4      # was 7: close segment after 4 quiet frames not 7
DYN_MIN_FRAMES   = 4      # was 5: attempt recognition on shorter segments
DYN_MAX_FRAMES   = 40     # was 50: tighter hard cap
COOLDOWN_FRAMES  = 6      # was 8: recover faster between signs


# ─── HELPERS ─────────────────────────────────────────────────────────────────

class LM:
    __slots__ = ("x", "y", "z")
    def __init__(self, d):
        self.x = d["x"]; self.y = d["y"]; self.z = d["z"]

def to_lm(lst):
    return [LM(d) for d in lst]


# ─── WEBSOCKET ────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Two queues: one for live caption words, one for TTS sentences.
    # Both are fed from background threads via call_soon_threadsafe.
    word_queue     = asyncio.Queue()
    sentence_queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def on_word(word):
        loop.call_soon_threadsafe(word_queue.put_nowait, word)

    def on_sentence(text, emotion):
        import time
        callback_time = time.time()
        print(f"[TTS TIMING] ⏱️  on_sentence() callback called at {callback_time:.3f} - text: '{text}', emotion: '{emotion}'")
        loop.call_soon_threadsafe(sentence_queue.put_nowait, (text, emotion))
        queue_put_time = time.time()
        print(f"[TTS TIMING] ⏱️  Sentence queued at {queue_put_time:.3f} (took {queue_put_time - callback_time:.3f}s to queue)")

    cleaner = StreamCleaner(on_word=on_word, on_sentence=on_sentence, sentence_pause=2.0)

    state = {
        # recording
        "rec_static":          False,
        "rec_dynamic":         False,
        "rec_name":            None,
        "rec_frames":          [],
        "rec_frames_other":    [],

        # static pipeline
        "vote_buf":            deque(maxlen=VOTE_WINDOW),

        # dynamic pipeline
        "dyn_capturing":       False,
        "dyn_frames":          [],
        "dyn_frames_other":    [],
        "dyn_quiet":           0,
        "prev_lm":             None,

        # shared cooldown
        "cooldown":            0,
        
        # emotion tracking
        "current_emotion":      "neutral",
    }

    async def send_words():
        """Sends live caption words immediately as they're cleaned."""
        try:
            while True:
                word = await word_queue.get()
                await websocket.send_text(json.dumps({
                    "type": "caption_word",
                    "word": word,
                }))
        except Exception:
            pass

    async def send_sentences():
        """Sends full cleaned sentences for TTS after natural pauses."""
        try:
            while True:
                import time
                chunk_start_time = time.time()
                text, emotion = await sentence_queue.get()
                queue_receive_time = time.time()
                print(f"[TTS TIMING] ⏱️  Chunk received from queue at {queue_receive_time:.3f} - text: '{text}', emotion: '{emotion}'")
                print(f"[TTS TIMING] ⏱️  Time since chunk completion: {queue_receive_time - chunk_start_time:.3f}s")
                
                # Synthesize speech with TTS service if available
                if tts_service:
                    print(f"[TTS TIMING] ⏱️  TTS service is available, starting synthesis...")
                    try:
                        tts_start_time = time.time()
                        print(f"[TTS TIMING] ⏱️  Calling synthesize_speech at {tts_start_time:.3f} with text='{text}', emotion='{emotion}'")
                        audio_data = await tts_service.synthesize_speech(
                            text=text,
                            emotion=emotion
                        )
                        tts_end_time = time.time()
                        tts_duration = tts_end_time - tts_start_time
                        print(f"[TTS TIMING] ⏱️  Synthesis completed at {tts_end_time:.3f} - took {tts_duration:.3f}s")
                        print(f"[TTS TIMING] ⏱️  Audio data length: {len(audio_data)} bytes")
                        
                        # Encode audio as base64 for transmission
                        encode_start_time = time.time()
                        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                        encode_end_time = time.time()
                        encode_duration = encode_end_time - encode_start_time
                        print(f"[TTS TIMING] ⏱️  Base64 encoding completed at {encode_end_time:.3f} - took {encode_duration:.3f}s")
                        print(f"[TTS TIMING] ⏱️  Base64 encoded audio length: {len(audio_b64)} characters")
                        
                        message = {
                            "type": "speak_sentence",
                            "text": text,
                            "emotion": emotion,
                            "audio": audio_b64,
                            "audio_format": "ogg"
                        }
                        send_start_time = time.time()
                        print(f"[TTS TIMING] ⏱️  Sending WebSocket message at {send_start_time:.3f} (audio field present: {bool(audio_b64)})")
                        await websocket.send_text(json.dumps(message))
                        send_end_time = time.time()
                        send_duration = send_end_time - send_start_time
                        total_duration = send_end_time - queue_receive_time
                        print(f"[TTS TIMING] ⏱️  WebSocket send completed at {send_end_time:.3f} - took {send_duration:.3f}s")
                        print(f"[TTS TIMING] ⏱️  TOTAL TIME from queue receive to send: {total_duration:.3f}s")
                        print(f"[TTS TIMING] ⏱️  Breakdown: TTS={tts_duration:.3f}s, Encode={encode_duration:.3f}s, Send={send_duration:.3f}s")
                    except Exception as e:
                        print(f"[TTS] TTS synthesis error: {e}")
                        import traceback
                        traceback.print_exc()
                        # Fallback: send text without audio
                        print(f"[TTS] Sending message WITHOUT audio as fallback")
                        await websocket.send_text(json.dumps({
                            "type": "speak_sentence",
                            "text": text,
                            "emotion": emotion,
                        }))
                else:
                    # No TTS service, just send text
                    print(f"[TTS] WARNING: TTS service is None! Sending message without audio")
                    await websocket.send_text(json.dumps({
                        "type": "speak_sentence",
                        "text": text,
                        "emotion": emotion,
                    }))
        except Exception as e:
            print(f"[TTS] Error in send_sentences: {e}")
            import traceback
            traceback.print_exc()

    word_task     = asyncio.create_task(send_words())
    sentence_task = asyncio.create_task(send_sentences())

    try:
        while True:
            raw      = await websocket.receive_text()
            data     = json.loads(raw)
            msg_type = data.get("type")

            # ── LANDMARK FRAME ─────────────────────────────────────────────
            if msg_type == "landmarks":
                frames   = data.get("landmarks", [])
                # Update current emotion if provided
                if "emotion" in data:
                    new_emotion = data.get("emotion", "neutral")
                    old_emotion = state["current_emotion"]
                    state["current_emotion"] = new_emotion
                    if new_emotion != old_emotion:
                        print(f"[Emotion] ✅ Emotion updated: '{old_emotion}' -> '{new_emotion}'")
                    # Debug: log all non-neutral emotions
                    if new_emotion != "neutral":
                        print(f"[Emotion] ✅✅✅ Received NON-NEUTRAL emotion '{new_emotion}' from frontend (raw data: {data.get('emotion')})")
                else:
                    # Debug: log when emotion is missing
                    print(f"[Emotion] ⚠️ WARNING: No 'emotion' field in landmarks message! Keys: {list(data.keys())}")
                    if state["current_emotion"] != "neutral":
                        print(f"[Emotion] Keeping current emotion: '{state['current_emotion']}'")
                response = {"type": "match"}

                if not frames:
                    state["vote_buf"].clear()
                    state["dyn_capturing"]    = False
                    state["dyn_frames"]       = []
                    state["dyn_frames_other"] = []
                    state["dyn_quiet"]        = 0
                    state["prev_lm"]          = None
                    await websocket.send_text(json.dumps(response))
                    continue

                curr_lm  = to_lm(frames[0])
                other_lm = to_lm(frames[1]) if len(frames) > 1 else None

                # ── STATIC RECORDING ──────────────────────────────────────
                if state["rec_static"]:
                    from recognizer import RECORDINGS_PER_SIGN
                    ok, count = recognizer.add_static_template(
                        state["rec_name"], curr_lm, other_lm)
                    state["rec_static"] = False
                    await websocket.send_text(json.dumps({
                        "type": "recording_done", "success": ok,
                        "name": state["rec_name"], "gesture_type": "static",
                        "count": count, "needed": RECORDINGS_PER_SIGN,
                        "ready": count >= RECORDINGS_PER_SIGN,
                    }))
                    state["prev_lm"] = curr_lm
                    continue

                # ── DYNAMIC RECORDING ─────────────────────────────────────
                if state["rec_dynamic"]:
                    state["rec_frames"].append(curr_lm)
                    state["rec_frames_other"].append(other_lm)
                    response["recording"]   = True
                    response["frame_count"] = len(state["rec_frames"])
                    await websocket.send_text(json.dumps(response))
                    state["prev_lm"] = curr_lm
                    continue

                # ── COOLDOWN ──────────────────────────────────────────────
                if state["cooldown"] > 0:
                    state["cooldown"] -= 1
                    state["prev_lm"]   = curr_lm
                    await websocket.send_text(json.dumps(response))
                    continue

                # ── VELOCITY ──────────────────────────────────────────────
                vel = 0.0
                if state["prev_lm"] is not None:
                    vel = fingertip_velocity(state["prev_lm"], curr_lm)
                state["prev_lm"] = curr_lm

                recognized = None
                confidence = 0.0

                # ── PIPELINE 1: STATIC ────────────────────────────────────
                s_sign, s_conf = recognizer.recognize_static(curr_lm, other_lm)
                state["vote_buf"].append(s_sign if s_sign else "__none__")

                if len(state["vote_buf"]) >= VOTE_WINDOW:
                    votes = [v for v in state["vote_buf"] if v != "__none__"]
                    if votes:
                        top, top_n = Counter(votes).most_common(1)[0]
                        ratio      = top_n / len(state["vote_buf"])
                        if ratio >= VOTE_THRESHOLD:
                            recognized = top
                            confidence = ratio
                            state["vote_buf"].clear()

                    if not recognized:
                        for _ in range(VOTE_WINDOW // 4):
                            if state["vote_buf"]:
                                state["vote_buf"].popleft()

                # ── PIPELINE 2: DYNAMIC ───────────────────────────────────
                if not recognized:
                    if not state["dyn_capturing"]:
                        if vel > DYN_ONSET_VEL:
                            # Clear stale static votes so pre-motion frames
                            # don't dilute the vote for a static sign held after motion
                            state["vote_buf"].clear()
                            state["dyn_capturing"]    = True
                            state["dyn_frames"]       = [curr_lm]
                            state["dyn_frames_other"] = [other_lm]
                            state["dyn_quiet"]        = 0
                    else:
                        state["dyn_frames"].append(curr_lm)
                        state["dyn_frames_other"].append(other_lm)

                        if vel < DYN_END_VEL:
                            state["dyn_quiet"] += 1
                        else:
                            state["dyn_quiet"] = 0

                        seg_len  = len(state["dyn_frames"])
                        seg_done = (
                            state["dyn_quiet"] >= DYN_END_FRAMES or
                            seg_len >= DYN_MAX_FRAMES
                        )

                        if seg_done:
                            if seg_len >= DYN_MIN_FRAMES:
                                trim    = max(DYN_MIN_FRAMES, seg_len - state["dyn_quiet"])
                                seg     = state["dyn_frames"][:trim]
                                seg_oth = state["dyn_frames_other"][:trim]
                                d_sign, d_conf = recognizer.recognize_dynamic(seg, seg_oth)
                                if d_sign:
                                    recognized = d_sign
                                    confidence = d_conf

                            state["dyn_capturing"]    = False
                            state["dyn_frames"]       = []
                            state["dyn_frames_other"] = []
                            state["dyn_quiet"]        = 0

                # ── EMIT ──────────────────────────────────────────────────
                if recognized:
                    # Always show the raw recognition badge on the camera feed
                    response["recognized"] = recognized
                    response["confidence"] = round(confidence, 2)
                    state["cooldown"]      = COOLDOWN_FRAMES
                    state["vote_buf"].clear()
                    state["dyn_capturing"]    = False
                    state["dyn_frames"]       = []
                    state["dyn_frames_other"] = []

                    # Feed raw word into cleaner with current emotion
                    # cleaned output comes back asynchronously via on_sentence
                    current_emotion = state["current_emotion"]
                    print(f"[Emotion] Pushing word '{recognized}' with emotion: '{current_emotion}'")
                    import time
                    chunk_complete_time = time.time()
                    print(f"[TTS TIMING] ⏱️  Chunk completed and pushed to cleaner at {chunk_complete_time:.3f}")
                    cleaner.push(recognized, current_emotion)

                await websocket.send_text(json.dumps(response))

            # ── RECORDING CONTROLS ─────────────────────────────────────────
            elif msg_type == "start_static_recording":
                state["rec_name"]   = data["name"]
                state["rec_static"] = True
                state["cooldown"]   = 0
                state["vote_buf"].clear()
                await websocket.send_text(json.dumps(
                    {"type": "recording_started", "gesture_type": "static"}))

            elif msg_type == "start_dynamic_recording":
                state["rec_name"]         = data["name"]
                state["rec_dynamic"]      = True
                state["rec_frames"]       = []
                state["rec_frames_other"] = []
                state["cooldown"]         = 0
                await websocket.send_text(json.dumps(
                    {"type": "recording_started", "gesture_type": "dynamic"}))

            elif msg_type == "stop_dynamic_recording":
                from recognizer import RECORDINGS_PER_SIGN
                frames     = state["rec_frames"]
                frames_oth = state["rec_frames_other"]
                ok, count  = recognizer.add_dynamic_template(
                    state["rec_name"], frames, frames_oth)
                state["rec_dynamic"]      = False
                state["rec_frames"]       = []
                state["rec_frames_other"] = []
                await websocket.send_text(json.dumps({
                    "type": "recording_done", "success": ok,
                    "name": state["rec_name"], "gesture_type": "dynamic",
                    "frame_count": len(frames), "count": count,
                    "needed": RECORDINGS_PER_SIGN,
                    "ready": count >= RECORDINGS_PER_SIGN,
                }))

            elif msg_type == "get_templates":
                await websocket.send_text(json.dumps(
                    {"type": "templates",
                     "templates": recognizer.get_all_templates()}))

            elif msg_type == "delete_template":
                ok = recognizer.delete_template(data["name"])
                await websocket.send_text(json.dumps(
                    {"type": "deleted", "name": data["name"], "success": ok}))

            elif msg_type == "rename_template":
                old_name = data.get("old_name")
                new_name = data.get("new_name", "").strip()
                if old_name and new_name and old_name in recognizer.templates:
                    if new_name not in recognizer.templates:
                        recognizer.templates[new_name] = recognizer.templates.pop(old_name)
                        recognizer.save_templates()
                        recognizer._rebuild_classifier()
                        await websocket.send_text(json.dumps({
                            "type": "renamed",
                            "old_name": old_name,
                            "new_name": new_name,
                        }))
                    else:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": f"'{new_name}' already exists",
                        }))

            elif msg_type == "reset_cleaner":
                # Presenter stopped — flush whatever is buffered immediately
                cleaner.force_flush()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WS error: {e}")
        import traceback; traceback.print_exc()
    finally:
        word_task.cancel()
        sentence_task.cancel()
        cleaner.reset()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)