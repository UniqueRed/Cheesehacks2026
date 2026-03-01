import asyncio
import json
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from recognizer import GestureRecognizer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

recognizer = GestureRecognizer()
connections = {}

MOTION_THRESHOLD = 0.015
SIGN_COOLDOWN_SECONDS = 2.0  # seconds before same sign can fire again

def has_motion(buffer):
    if len(buffer) < 4:
        return False
    wrist_positions = [(frame[0]['x'], frame[0]['y']) for frame in buffer]
    total_movement = sum(
        ((wrist_positions[i][0] - wrist_positions[i-1][0])**2 +
         (wrist_positions[i][1] - wrist_positions[i-1][1])**2) ** 0.5
        for i in range(1, len(wrist_positions))
    )
    return total_movement > MOTION_THRESHOLD


class LandmarkProxy:
    def __init__(self, lm_dict):
        self.x = lm_dict["x"]
        self.y = lm_dict["y"]
        self.z = lm_dict["z"]


def dicts_to_landmarks(lm_list):
    return [LandmarkProxy(d) for d in lm_list]


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    conn_id = id(websocket)
    connections[conn_id] = {
        "recording_static": False,
        "recording_dynamic": False,
        "dynamic_frames": [],
        "recording_name": None,
        "last_recognized": None,
        "debounce_count": 0,
        "required_debounce": 8,
        "dynamic_buffer": [],
        "dynamic_buffer_other": [],
        "dynamic_buffer_size": 20,
        "dynamic_frames_other": [],
        "sign_last_fired": {},  # sign_name -> timestamp
    }
    state = connections[conn_id]

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "landmarks":
                landmarks_data = data.get("landmarks", [])
                response = {"type": "match"}

                if landmarks_data:
                    primary = dicts_to_landmarks(landmarks_data[0])
                    other_hand = dicts_to_landmarks(landmarks_data[1]) if len(landmarks_data) > 1 else None

                    state["dynamic_buffer"].append(landmarks_data[0])
                    state["dynamic_buffer_other"].append(landmarks_data[1] if len(landmarks_data) > 1 else None)
                    if len(state["dynamic_buffer"]) > state["dynamic_buffer_size"]:
                        state["dynamic_buffer"].pop(0)
                        state["dynamic_buffer_other"].pop(0)

                    if state["recording_static"]:
                        success, count = recognizer.add_static_template(state["recording_name"], primary, other_hand)
                        state["recording_static"] = False
                        from recognizer import RECORDINGS_PER_SIGN
                        await websocket.send_text(json.dumps({
                            "type": "recording_done",
                            "success": success,
                            "name": state["recording_name"],
                            "gesture_type": "static",
                            "count": count,
                            "needed": RECORDINGS_PER_SIGN,
                            "ready": count >= RECORDINGS_PER_SIGN
                        }))
                        continue

                    if state["recording_dynamic"]:
                        state["dynamic_frames"].append(landmarks_data[0])
                        state["dynamic_frames_other"].append(landmarks_data[1] if len(landmarks_data) > 1 else None)
                        response["recording"] = True
                        response["frame_count"] = len(state["dynamic_frames"])

                    if not state["recording_static"] and not state["recording_dynamic"]:
                        sign, confidence = recognizer.recognize_static(primary, other_hand)

                        if not sign and len(state["dynamic_buffer"]) >= 10:
                            # Only attempt dynamic matching if there's actual motion
                            if has_motion(state["dynamic_buffer"]):
                                dynamic_lm = [dicts_to_landmarks(f) for f in state["dynamic_buffer"]]
                                other_lm = [dicts_to_landmarks(f) if f else None for f in state["dynamic_buffer_other"]]
                                sign, confidence = recognizer.recognize_dynamic(dynamic_lm, other_lm)

                        if sign:
                            if sign == state["last_recognized"]:
                                state["debounce_count"] += 1
                            else:
                                state["debounce_count"] = 1
                                state["last_recognized"] = sign

                            if state["debounce_count"] == state["required_debounce"]:
                                now = time.time()
                                last_fired = state["sign_last_fired"].get(sign, 0)
                                if now - last_fired >= SIGN_COOLDOWN_SECONDS:
                                    response["recognized"] = sign
                                    response["confidence"] = confidence
                                    state["sign_last_fired"][sign] = now
                                state["dynamic_buffer"] = []
                                state["dynamic_buffer_other"] = []
                                state["debounce_count"] = 0
                                state["last_recognized"] = None
                        else:
                            state["last_recognized"] = None
                            state["debounce_count"] = 0

                await websocket.send_text(json.dumps(response))

            elif msg_type == "start_static_recording":
                state["recording_name"] = data["name"]
                state["recording_static"] = True
                await websocket.send_text(json.dumps({"type": "recording_started", "gesture_type": "static"}))

            elif msg_type == "start_dynamic_recording":
                state["recording_name"] = data["name"]
                state["recording_dynamic"] = True
                state["dynamic_frames"] = []
                await websocket.send_text(json.dumps({"type": "recording_started", "gesture_type": "dynamic"}))

            elif msg_type == "stop_dynamic_recording":
                frames = state["dynamic_frames"]
                frames_other = state["dynamic_frames_other"]
                lm_sequences = [dicts_to_landmarks(f) for f in frames]
                other_sequences = [dicts_to_landmarks(f) if f else None for f in frames_other]
                success, count = recognizer.add_dynamic_template(state["recording_name"], lm_sequences, other_sequences)
                state["recording_dynamic"] = False
                state["dynamic_frames"] = []
                state["dynamic_frames_other"] = []
                from recognizer import RECORDINGS_PER_SIGN
                await websocket.send_text(json.dumps({
                    "type": "recording_done",
                    "success": success,
                    "name": state["recording_name"],
                    "gesture_type": "dynamic",
                    "frame_count": len(frames),
                    "count": count,
                    "needed": RECORDINGS_PER_SIGN,
                    "ready": count >= RECORDINGS_PER_SIGN
                }))
                await websocket.send_text(json.dumps({
                    "type": "recording_done",
                    "success": success,
                    "name": state["recording_name"],
                    "gesture_type": "dynamic",
                    "frame_count": len(frames)
                }))

            elif msg_type == "get_templates":
                templates = recognizer.get_all_templates()
                await websocket.send_text(json.dumps({"type": "templates", "templates": templates}))

            elif msg_type == "delete_template":
                success = recognizer.delete_template(data["name"])
                await websocket.send_text(json.dumps({"type": "deleted", "name": data["name"], "success": success}))

    except WebSocketDisconnect:
        del connections[conn_id]
    except Exception as e:
        print(f"Error: {e}")
        if conn_id in connections:
            del connections[conn_id]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)