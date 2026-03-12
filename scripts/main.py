import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque, Counter
from scipy.spatial import ConvexHull
import pandas as pd

RIGHT_CORNER_OUTER = 61
RIGHT_UPPER_OUTER = 74
TOP_RIGHT_OUTER = 72
TOP_LEFT_OUTER = 302
LEFT_UPPER_OUTER = 304
LEFT_CORENER_OUTER = 291
LEFT_LOWER_OUTER = 320
LOW_LEFT_OUTER = 315
LOW_RIGHT_OUTER = 85
RIGHT_LOWER_OUTER = 90

RIGHT_CORNER_INNER = 78
RIGHT_UPPER_INNER = 80
TOP_RIGHT_INNER = 82
TOP_LEFT_INNER = 312
LEFT_UPPER_INNER = 310
LEFT_CORENER_INNER = 308
LEFT_LOWER_INNER = 318
LOW_LEFT_INNER = 317
LOW_RIGHT_INNER = 87
RIGHT_LOWER_INNER = 88

UPPER_LIP_INNER = 13
LOWER_LIP_INNER = 14

UPPER_LIP_OUTER = 0
LOWER_LIP_OUTER = 17

FACE_LEFT = 234
FACE_RIGHT = 454


def calculate_distance(p1, p2, w, h):
    return np.linalg.norm(np.array([float(p1.x * w), float(p1.y * h)]) - np.array([float(p2.x * w), float(p2.y * h)]))

def calculate_angle(p1, vertex, p2, w, h):
    v1 = np.array([p1.x * w, p1.y * h])
    v_vertex = np.array([vertex.x * w, vertex.y * h])
    v2 = np.array([p2.x * w, p2.y * h])
    ba = v1 - v_vertex
    bc = v2 - v_vertex
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

model = joblib.load("mlp_model.pkl")
scaler = joblib.load("scaler.pkl")

n_frames = 5
feature_buffer = deque(maxlen=n_frames)
viseme_results = []

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5)

def get_viseme_list(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        success, image = cap.read()

        if not success: 
            break

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w, _ = image.shape

            face_l = face_landmarks.landmark[FACE_LEFT]
            face_r = face_landmarks.landmark[FACE_RIGHT]
            
            face_width = calculate_distance(face_l, face_r, w, h)

            outer_points_raw = [
            face_landmarks.landmark[RIGHT_CORNER_OUTER], face_landmarks.landmark[RIGHT_UPPER_OUTER],
            face_landmarks.landmark[TOP_RIGHT_OUTER], face_landmarks.landmark[TOP_LEFT_OUTER],
            face_landmarks.landmark[LEFT_UPPER_OUTER], face_landmarks.landmark[LEFT_CORENER_OUTER],
            face_landmarks.landmark[LEFT_LOWER_OUTER], face_landmarks.landmark[LOW_LEFT_OUTER],
            face_landmarks.landmark[LOW_RIGHT_OUTER], face_landmarks.landmark[RIGHT_LOWER_OUTER]
            ]

            cx = sum(p.x for p in outer_points_raw) / 10.0
            cy = sum(p.y for p in outer_points_raw) / 10.0

            shape_features = []
            for p in outer_points_raw:
                dist = np.linalg.norm(np.array([p.x * w, p.y * h]) - np.array([cx * w, cy * h]))
                shape_features.append(dist / face_width)

            opening_vertical = calculate_distance(face_landmarks.landmark[UPPER_LIP_INNER], 
                                          face_landmarks.landmark[LOWER_LIP_INNER], w, h)
            opening_horizontal = calculate_distance(face_landmarks.landmark[LEFT_CORENER_INNER], face_landmarks.landmark[RIGHT_CORNER_INNER], w, h)

            
            mar = opening_vertical / (opening_horizontal + 1e-6)

            angle_l = calculate_angle(
                face_landmarks.landmark[TOP_LEFT_OUTER],
                face_landmarks.landmark[LEFT_CORENER_OUTER],
                face_landmarks.landmark[LOW_LEFT_OUTER],
                w, h
            )

            angle_r = calculate_angle(
                face_landmarks.landmark[TOP_RIGHT_OUTER],
                face_landmarks.landmark[RIGHT_CORNER_OUTER],
                face_landmarks.landmark[LOW_RIGHT_OUTER],
                w, h
            )

            nose_tip = face_landmarks.landmark[1]
            chin = face_landmarks.landmark[152]

            jaw_drop = calculate_distance(nose_tip, chin, w, h) / face_width

            current_frame_features = shape_features + [mar, angle_l, angle_r, jaw_drop]

            input_df = pd.DataFrame([current_frame_features], columns=[0,1,2,3,4,5,6,7,8,9,10,11,12,13])
            
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            
            viseme_results.append(int(prediction))
    cap.release()
    return viseme_results

all_visemes = get_viseme_list("/home/srki/Documents/OMV/spk08and14/spk08/ser/video_a_anonymized/spk08_005.mp4")

summary = [all_visemes[i] for i in range(len(all_visemes)) if i == 0 or all_visemes[i] != all_visemes[i-1]]
print("Sequence of Visemes:", summary)