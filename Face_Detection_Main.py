import cv2
import mediapipe as mp
import numpy as np
from Emotion_Analysis import emotion_Analysis
from Visualize import save_score, plot_emotion_history_to_cv2img
from Preprocessing import preprocess_input_for_miniXception

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("웹캠이 열리지 않습니다.")
    exit()

frame_counter = 0
graph_img = None

def crop_face_dynamic_size(image, detection, scale_factor=2.0):
    h, w, _ = image.shape
    kps = detection.location_data.relative_keypoints
    try:
        left_eye = np.array([kps[0].x * w, kps[0].y * h])
        right_eye = np.array([kps[1].x * w, kps[1].y * h])
        nose = np.array([kps[2].x * w, kps[2].y * h])
        mouth_l = np.array([kps[3].x * w, kps[3].y * h])
        mouth_r = np.array([kps[4].x * w, kps[4].y * h])
    except:
        return None, None

    center = np.mean([left_eye, right_eye, nose, mouth_l, mouth_r], axis=0)
    d_eye = np.linalg.norm(left_eye - right_eye)
    d_mouth = np.linalg.norm(mouth_l - mouth_r)
    d_eye_nose = np.linalg.norm((left_eye + right_eye)/2 - nose)
    base_size = max(d_eye, d_mouth, d_eye_nose)
    box_size = int(base_size * scale_factor)
    x1 = max(0, int(center[0] - box_size / 2))
    y1 = max(0, int(center[1] - box_size / 2))
    x2 = min(w, int(center[0] + box_size / 2))
    y2 = min(h, int(center[1] + box_size / 2))
    return image[y1:y2, x1:x2], (x1, y1, x2, y2)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    print("얼굴 검출기 실행")

    frame_rgb = None
    results = None

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("[프레임 수신 실패]")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 3프레임마다만 연산
        if frame_counter % 3 == 0:
            results = face_detection.process(frame_rgb)

            if results.detections:
                detection = results.detections[0]
                face_crop, _ = crop_face_dynamic_size(frame, detection)

                if face_crop is not None:
                    input_tensor = preprocess_input_for_miniXception(face_crop)
                    emotion_scores = emotion_Analysis(input_tensor)

                    if emotion_scores:
                        save_score(emotion_scores, frame_counter)

                        graph_img = plot_emotion_history_to_cv2img()
                        if graph_img is not None:
                            cv2.imshow('Emotion Graph', graph_img)

                        info_img = np.ones((150, 300, 3), dtype=np.uint8) * 255
                        for idx, (emotion, score) in enumerate(emotion_scores.items()):
                            cv2.putText(info_img, f"{emotion}: {score:.2f}", (10, 25 + idx * 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.imshow('Emotion Probabilities', info_img)

        # 항상 보여주는 기본 영상
        cv2.imshow('Mediapipe Face Detection', frame)
        frame_counter += 1

        if cv2.waitKey(1) & 0xFF == 27:
            print("[ESC 눌림 → 종료]")
            break

cap.release()
cv2.destroyAllWindows()
