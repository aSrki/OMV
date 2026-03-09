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
testing = True 
data_cnt = 8

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

            outer_points = [
                [face_landmarks.landmark[RIGHT_CORNER_OUTER].x, face_landmarks.landmark[RIGHT_CORNER_OUTER].y],
                [face_landmarks.landmark[RIGHT_UPPER_OUTER].x, face_landmarks.landmark[RIGHT_UPPER_OUTER].y],
                [face_landmarks.landmark[TOP_RIGHT_OUTER].x, face_landmarks.landmark[TOP_RIGHT_OUTER].y],
                [face_landmarks.landmark[TOP_LEFT_OUTER].x, face_landmarks.landmark[TOP_LEFT_OUTER].y],
                [face_landmarks.landmark[LEFT_UPPER_OUTER].x, face_landmarks.landmark[LEFT_UPPER_OUTER].y],
                [face_landmarks.landmark[LEFT_CORENER_OUTER].x, face_landmarks.landmark[LEFT_CORENER_OUTER].y],
                [face_landmarks.landmark[LEFT_LOWER_OUTER].x, face_landmarks.landmark[LEFT_LOWER_OUTER].y],
                [face_landmarks.landmark[LOW_LEFT_OUTER].x, face_landmarks.landmark[LOW_LEFT_OUTER].y],
                [face_landmarks.landmark[LOW_RIGHT_OUTER].x, face_landmarks.landmark[LOW_RIGHT_OUTER].y],
                [face_landmarks.landmark[RIGHT_LOWER_OUTER].x, face_landmarks.landmark[RIGHT_LOWER_OUTER].y]
            ]

            inner_points = [
                [face_landmarks.landmark[RIGHT_CORNER_INNER].x, face_landmarks.landmark[RIGHT_CORNER_INNER].y],
                [face_landmarks.landmark[RIGHT_UPPER_INNER].x, face_landmarks.landmark[RIGHT_UPPER_INNER].y],
                [face_landmarks.landmark[TOP_RIGHT_INNER].x, face_landmarks.landmark[TOP_RIGHT_INNER].y],
                [face_landmarks.landmark[TOP_LEFT_INNER].x, face_landmarks.landmark[TOP_LEFT_INNER].y],
                [face_landmarks.landmark[LEFT_UPPER_INNER].x, face_landmarks.landmark[LEFT_UPPER_INNER].y],
                [face_landmarks.landmark[LEFT_CORENER_INNER].x, face_landmarks.landmark[LEFT_CORENER_INNER].y],
                [face_landmarks.landmark[LEFT_LOWER_INNER].x, face_landmarks.landmark[LEFT_LOWER_INNER].y],
                [face_landmarks.landmark[LOW_LEFT_INNER].x, face_landmarks.landmark[LOW_LEFT_INNER].y],
                [face_landmarks.landmark[LOW_RIGHT_INNER].x, face_landmarks.landmark[LOW_RIGHT_INNER].y],
                [face_landmarks.landmark[RIGHT_LOWER_INNER].x, face_landmarks.landmark[RIGHT_LOWER_INNER].y]
            ]

            outer_area = ConvexHull(outer_points)
            inner_area = ConvexHull(inner_points)

            # face_l = face_landmarks.landmark[FACE_LEFT]
            # face_r = face_landmarks.landmark[FACE_RIGHT]

            upper_lip_inner = face_landmarks.landmark[UPPER_LIP_INNER]
            lower_lip_inner = face_landmarks.landmark[LOWER_LIP_INNER]

            left_inner = face_landmarks.landmark[LEFT_CORENER_INNER]
            right_inner = face_landmarks.landmark[RIGHT_CORNER_INNER]

            # face_width = calculate_distance(face_l, face_r, w, h)
            opening_vertical = calculate_distance(upper_lip_inner, lower_lip_inner, w, h)
            opening_horizontal = calculate_distance(left_inner, right_inner, w, h)
            # l1 = calculate_distance(righ_corner, right_upper, w, h)/face_width
            # l2 = calculate_distance(right_upper, top_right, w, h)/face_width
            # l3 = calculate_distance(top_right, top_left, w, h)/face_width
            # l4 = calculate_distance(top_left, left_upper, w, h)/face_width
            # l5 = calculate_distance(left_upper, left_corner, w, h)/face_width
            # l6 = calculate_distance(left_corner, left_lower, w, h)/face_width
            # l7 = calculate_distance(left_lower, low_left, w, h)/face_width
            # l8 = calculate_distance(low_left, low_right, w, h)/face_width
            # l9 = calculate_distance(low_right, right_lower, w, h)/face_width
            # l10 = calculate_distance(right_lower, righ_corner, w, h)/face_width


            for annotation in annotations:
                if annotation[1] >= frame_cnt >= annotation[0]:
                    # features.append([l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, opening, annotation[2]])
                    features.append([outer_area.volume, outer_area.area, inner_area.volume, inner_area.area, outer_area.volume/inner_area.volume, outer_area.area/inner_area.area, opening_horizontal/opening_vertical, annotation[2]])
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