from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
import base64
import mediapipe as mp
from fastapi.middleware.cors import CORSMiddleware

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FrameRequest(BaseModel):
    frame: str  

@app.post("/draw_landmarks/")
async def draw_landmarks(request: FrameRequest):
    try:
        # Decode Base64 to numpy array
        frame_data = base64.b64decode(request.frame)
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        landmarks_list = []
        bounding_box = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                hand_landmarks_points = []
                for landmark in hand_landmarks.landmark:
                    h, w, _ = frame.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    z = landmark.z  
                    hand_landmarks_points.append([x, y, z])

                landmarks_list.append(hand_landmarks_points)

                x_min = min(point[0] for point in hand_landmarks_points)
                y_min = min(point[1] for point in hand_landmarks_points)
                x_max = max(point[0] for point in hand_landmarks_points)
                y_max = max(point[1] for point in hand_landmarks_points)
                bounding_box = [x_min, y_min, x_max - x_min, y_max - y_min]

        _, buffer = cv2.imencode('.jpg', frame)
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        return JSONResponse(content={
            "frame": encoded_image,
            "landmarks": landmarks_list if landmarks_list else None,
            "bounding_box": bounding_box if bounding_box else None
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
