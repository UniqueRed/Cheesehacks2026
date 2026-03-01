"""
main.py — SignSpeak WebSocket backend.

Architecture:
  Static signs:  Voting window (16 frames, 72% agreement) → SVM classifier.
  Dynamic signs: Motion-segmented capture → DTW on finger-state sequences.
  Cleaning:      StreamCleaner buffers raw words, flushes cleaned chunks
                 after 2s of silence or when presenter stops.
"""

import json
import numpy as np
from collections import deque, Counter
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from recognizer import GestureRecognizer
from features import extract_dynamic_frame, fingertip_velocity
from cleaner import StreamCleaner

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

recognizer = GestureRecognizer()

# ─── TUNING ───────────────────────────────────────────────────────────────────
VOTE_WINDOW      = 16
VOTE_THRESHOLD   = 0.72
DYN_ONSET_VEL    = 0.018
DYN_END_VEL      = 0.010
DYN_END_FRAMES   = 7
DYN_MIN_FRAMES   = 5
DYN_MAX_FRAMES   = 50
COOLDOWN_FRAMES  = 8


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

    # Cleaned chunks produced by the background cleaner thread
    # are appended here and sent on the next frame tick.
    pending_chunks = []

    def on_cleaned(text):
        pending_chunks.append(text)

    cleaner = StreamCleaner(on_flushed=on_cleaned, flush_after=2.0)

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
    }

    try:
        while True:
            raw      = await websocket.receive_text()
            data     = json.loads(raw)
            msg_type = data.get("type")

            # ── FLUSH PENDING CLEANED CHUNKS ──────────────────────────────
            # Check on every tick so chunks get sent promptly.
            for chunk in list(pending_chunks):
                await websocket.send_text(json.dumps({
                    "type": "cleaned_chunk",
                    "text": chunk,
                }))
            pending_chunks.clear()

            # ── LANDMARK FRAME ─────────────────────────────────────────────
            if msg_type == "landmarks":
                frames   = data.get("landmarks", [])
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

                    # Feed raw word into cleaner — cleaned output comes
                    # back asynchronously via on_cleaned → pending_chunks
                    cleaner.push(recognized)

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
        cleaner.reset()
    except Exception as e:
        print(f"WS error: {e}")
        import traceback; traceback.print_exc()
        cleaner.reset()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)