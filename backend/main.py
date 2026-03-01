"""
main.py — SignSpeak WebSocket backend.

Architecture:
  Static signs:  Voting window (16 frames, 72% agreement) → SVM classifier.
  Dynamic signs: Motion-segmented capture → DTW on finger-state sequences.

Two pipelines run in parallel. Once either fires, a short cooldown prevents
double-firing. Dynamic pipeline resets after every fired or abandoned segment.
"""

import json
import numpy as np
from collections import deque, Counter
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from recognizer import GestureRecognizer
from features import extract_dynamic_frame, fingertip_velocity

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

recognizer = GestureRecognizer()

# ─── TUNING ───────────────────────────────────────────────────────────────────
# Static pipeline
VOTE_WINDOW      = 16     # frames in voting window (~1s at 15fps)
VOTE_THRESHOLD   = 0.72   # fraction of frames that must agree

# Dynamic pipeline — motion segmentation
DYN_ONSET_VEL    = 0.018  # fingertip velocity to start capturing
DYN_END_VEL      = 0.010  # fingertip velocity to count as "quiet"
DYN_END_FRAMES   = 7      # consecutive quiet frames → segment done
DYN_MIN_FRAMES   = 5      # minimum frames to attempt classification
DYN_MAX_FRAMES   = 50     # hard cap on segment length (~3.3s at 15fps)

# Shared post-fire cooldown
COOLDOWN_FRAMES  = 8      # frames to ignore after any recognition fires


# ─── HELPERS ─────────────────────────────────────────────────────────────────

class LM:
    """Lightweight landmark proxy."""
    __slots__ = ("x", "y", "z")
    def __init__(self, d):
        self.x = d["x"]; self.y = d["y"]; self.z = d["z"]

def to_lm(lst):
    return [LM(d) for d in lst]

def make_state():
    return {
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
        "dyn_frames":          [],          # landmark objects
        "dyn_frames_other":    [],
        "dyn_quiet":           0,           # consecutive quiet frames
        "prev_lm":             None,        # previous frame landmarks (for velocity)

        # shared cooldown
        "cooldown":            0,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    state = make_state()

    try:
        while True:
            raw      = await websocket.receive_text()
            data     = json.loads(raw)
            msg_type = data.get("type")

            # ── LANDMARK FRAME ─────────────────────────────────────────────
            if msg_type == "landmarks":
                frames   = data.get("landmarks", [])
                response = {"type": "match"}

                if not frames:
                    # Hand left — full reset
                    state["vote_buf"].clear()
                    state["dyn_capturing"] = False
                    state["dyn_frames"]    = []
                    state["dyn_frames_other"] = []
                    state["dyn_quiet"]     = 0
                    state["prev_lm"]       = None
                    await websocket.send_text(json.dumps(response))
                    continue

                curr_lm    = to_lm(frames[0])
                other_lm   = to_lm(frames[1]) if len(frames) > 1 else None
                raw_frame  = frames[0]
                raw_other  = frames[1] if len(frames) > 1 else None

                # ── STATIC RECORDING (one snapshot) ───────────────────────
                if state["rec_static"]:
                    from recognizer import RECORDINGS_PER_SIGN
                    ok, count = recognizer.add_static_template(
                        state["rec_name"], curr_lm, other_lm)
                    state["rec_static"] = False
                    await websocket.send_text(json.dumps({
                        "type":    "recording_done",
                        "success": ok,
                        "name":    state["rec_name"],
                        "gesture_type": "static",
                        "count":   count,
                        "needed":  RECORDINGS_PER_SIGN,
                        "ready":   count >= RECORDINGS_PER_SIGN,
                    }))
                    state["prev_lm"] = curr_lm
                    continue

                # ── DYNAMIC RECORDING (collect frames) ────────────────────
                if state["rec_dynamic"]:
                    state["rec_frames"].append(curr_lm)
                    state["rec_frames_other"].append(other_lm)
                    response["recording"]    = True
                    response["frame_count"]  = len(state["rec_frames"])
                    await websocket.send_text(json.dumps(response))
                    state["prev_lm"] = curr_lm
                    continue

                # ── COOLDOWN ──────────────────────────────────────────────
                if state["cooldown"] > 0:
                    state["cooldown"] -= 1
                    state["prev_lm"]   = curr_lm
                    await websocket.send_text(json.dumps(response))
                    continue

                # ── COMPUTE VELOCITY ───────────────────────────────────────
                vel = 0.0
                if state["prev_lm"] is not None:
                    vel = fingertip_velocity(state["prev_lm"], curr_lm)
                state["prev_lm"] = curr_lm

                recognized = None
                confidence = 0.0

                # ════════════════════════════════════════════════════════
                # PIPELINE 1: STATIC — voting window over SVM predictions
                # ════════════════════════════════════════════════════════
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
                        # Slide: drop oldest quarter so window advances smoothly
                        for _ in range(VOTE_WINDOW // 4):
                            if state["vote_buf"]:
                                state["vote_buf"].popleft()

                # ════════════════════════════════════════════════════════
                # PIPELINE 2: DYNAMIC — motion-segmented DTW
                # ════════════════════════════════════════════════════════
                if not recognized:
                    if not state["dyn_capturing"]:
                        if vel > DYN_ONSET_VEL:
                            # Motion started — begin segment
                            state["dyn_capturing"]    = True
                            state["dyn_frames"]       = [curr_lm]
                            state["dyn_frames_other"] = [other_lm]
                            state["dyn_quiet"]        = 0
                    else:
                        # Currently capturing
                        state["dyn_frames"].append(curr_lm)
                        state["dyn_frames_other"].append(other_lm)

                        if vel < DYN_END_VEL:
                            state["dyn_quiet"] += 1
                        else:
                            state["dyn_quiet"] = 0

                        seg_len   = len(state["dyn_frames"])
                        seg_done  = (
                            state["dyn_quiet"] >= DYN_END_FRAMES or
                            seg_len >= DYN_MAX_FRAMES
                        )

                        if seg_done:
                            if seg_len >= DYN_MIN_FRAMES:
                                # Trim trailing quiet frames
                                trim     = max(DYN_MIN_FRAMES,
                                               seg_len - state["dyn_quiet"])
                                seg      = state["dyn_frames"][:trim]
                                seg_oth  = state["dyn_frames_other"][:trim]

                                d_sign, d_conf = recognizer.recognize_dynamic(
                                    seg, seg_oth)
                                if d_sign:
                                    recognized = d_sign
                                    confidence = d_conf

                            # Reset segment
                            state["dyn_capturing"]    = False
                            state["dyn_frames"]       = []
                            state["dyn_frames_other"] = []
                            state["dyn_quiet"]        = 0

                # ── EMIT ──────────────────────────────────────────────────
                if recognized:
                    response["recognized"] = recognized
                    response["confidence"] = round(confidence, 2)
                    state["cooldown"]      = COOLDOWN_FRAMES
                    state["vote_buf"].clear()
                    state["dyn_capturing"]    = False
                    state["dyn_frames"]       = []
                    state["dyn_frames_other"] = []

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
                state["rec_name"]          = data["name"]
                state["rec_dynamic"]       = True
                state["rec_frames"]        = []
                state["rec_frames_other"]  = []
                state["cooldown"]          = 0
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
                    "type":         "recording_done",
                    "success":      ok,
                    "name":         state["rec_name"],
                    "gesture_type": "dynamic",
                    "frame_count":  len(frames),
                    "count":        count,
                    "needed":       RECORDINGS_PER_SIGN,
                    "ready":        count >= RECORDINGS_PER_SIGN,
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
                            "type": "renamed", "old_name": old_name, "new_name": new_name
                        }))
                    else:
                        await websocket.send_text(json.dumps({
                            "type": "error", "message": f"'{new_name}' already exists"
                        }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WS error: {e}")
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)