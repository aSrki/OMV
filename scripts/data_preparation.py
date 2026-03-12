import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from collections import Counter
from scipy.spatial import ConvexHull

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

VISEM_MAP = {
    'p': 'V0', 'b': 'V0', 'm': 'V0',
    'f': 'V1', 'v': 'V1',
    'a': 'V2',
    'e': 'V3', 'i' : 'V3',
    'o': 'V4',
    'u' : 'V5',
    't': 'V6', 'd': 'V6', 's': 'V6', 'z': 'V6', 'n': 'V6', 'c': 'V6',
    'č' : 'V7', 'ć' : 'V7', 'dž' : 'V7', 'đ' : 'V7', 'š' : 'V7', 'ž' : 'V7',
    'k' : 'V8', 'g' : 'V8', 'h' : 'V8',
    'sil' : 'V9',
    'j' : 'V10', 'ǉ' : 'V10', 'ǌ' : 'V10', 'r':'V10', 'l':'V10'
}

def sliding_window(features, n_frames, data_len = 12):
    windowed_features = []
    
    for i in range(len(features) - (n_frames - 1)):
        window_labels = []
        windowed_feature = np.zeros(data_len)
        for j in range(n_frames):
            windowed_feature[0:data_len-1] += np.array(features[i+j][0:data_len-1])
            window_labels.append(features[i+j][data_len-1])

        counts = Counter(window_labels)
        most_common_element, _ = counts.most_common(1)[0]
        windowed_feature[data_len-1] = int(most_common_element.lstrip('vV'))

        for i in range(data_len-1):
            windowed_feature[i] /= n_frames

        windowed_features.append(windowed_feature.tolist())

    return windowed_features

def calculate_angle(p1, vertex, p2, w, h):
    v1 = np.array([p1.x * w, p1.y * h])
    v_vertex = np.array([vertex.x * w, vertex.y * h])
    v2 = np.array([p2.x * w, p2.y * h])

    ba = v1 - v_vertex
    bc = v2 - v_vertex

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)

def load_annotations(file_path, fps):
    annotations = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3: 
                continue
            start_t = float(parts[0])
            end_t = float(parts[1])
            fonem = parts[2]
            
            visem = VISEM_MAP.get(fonem)

            if not fonem in VISEM_MAP:
                print(fonem)
                print(f"GRESKA {visem}")
                print(file_path)
                continue
            
            start_frame = int(start_t * fps)
            end_frame = int(end_t * fps)
            
            annotations.append([start_frame, end_frame, visem])
    return annotations

def calculate_distance(p1, p2, w, h):
    return np.linalg.norm(np.array([float(p1.x * w), float(p1.y * h)]) - np.array([float(p2.x * w), float(p2.y * h)]))

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

features = []

fps = 100
testing = False
data_cnt = 15

for i in range(110):
    if i > 29 and i < 60:
        continue

    if testing:
        if not i in [5, 15, 25, 60, 70, 80, 90, 100]:
            continue
    else:
        if i in [5, 15, 25, 60, 70, 80, 90, 100]:
            continue

    annotations = load_annotations(f"labels_08_aligned/labels 08 srp/spk08_{i:03}.txt", fps)
    cap = cv2.VideoCapture(f"spk08and14/spk08/ser/video_a_anonymized/spk08_{i:03}.mp4") 
    frame_cnt = 0

    print(f'Processing video {i}')

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

            outer_coords = np.array([[p.x, p.y] for p in outer_points_raw])
            outer_hull = ConvexHull(outer_coords)

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

            for annotation in annotations:
                if annotation[1] >= frame_cnt >= annotation[0]:
                    features.append(current_frame_features +  [annotation[2]])
            frame_cnt += 1                

for i in [1,3,5,7]:

    final_features = []
    final_labels = []

    n_frames = i

    features_windowed = sliding_window(features, n_frames, data_cnt)

    for feature in features_windowed:
        final_features.append(feature[0:data_cnt-1])
        final_labels.append(feature[data_cnt-1])

    df = pd.DataFrame(final_features)
    df['label'] = final_labels
    testing_str = "testing" if testing else "training"
    df.to_csv(f'{testing_str}_data_a_{n_frames}.csv', index=False)

    print('DONE')