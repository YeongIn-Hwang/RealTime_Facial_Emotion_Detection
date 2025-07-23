#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import mediapipe as mp
import numpy as np
from Emotion_Analysis import emotion_Analysis

mp_face_detection = mp.solutions.face_detection # 얼굴 검출 모델 준비
mp_drawing = mp.solutions.drawing_utils # 얼굴 검출된 부분에 사각형 유틸 추가

cap = cv2.VideoCapture(0) #0번 장치로 기본 카메라를 오픈

if not (cap.isOpened()):
    print("웹캠이 열리지 않습니다.")
    exit()

def crop_face_dynamic_size(image, detection, scale_factor=2.0):
    
    h, w, _ = image.shape
    kps = detection.location_data.relative_keypoints

    # 눈(0,1), 코(2), 입(3,4)
    try:
        left_eye = np.array([kps[0].x * w, kps[0].y * h])
        right_eye = np.array([kps[1].x * w, kps[1].y * h])
        nose = np.array([kps[2].x * w, kps[2].y * h])
        mouth_l = np.array([kps[3].x * w, kps[3].y * h])
        mouth_r = np.array([kps[4].x * w, kps[4].y * h])
    except:
        return None, None

    # 중심점: 눈 + 코 + 입 평균
    center = np.mean([left_eye, right_eye, nose, mouth_l, mouth_r], axis=0)

    # 거리 기반 크기 계산
    d_eye = np.linalg.norm(left_eye - right_eye)
    d_mouth = np.linalg.norm(mouth_l - mouth_r)
    d_eye_nose = np.linalg.norm((left_eye + right_eye)/2 - nose)

    base_size = max(d_eye, d_mouth, d_eye_nose)
    box_size = int(base_size * scale_factor)

    # 좌표 계산
    x1 = int(center[0] - box_size / 2)
    y1 = int(center[1] - box_size / 2)
    x2 = int(center[0] + box_size / 2)
    y2 = int(center[1] + box_size / 2)

    # 이미지 범위 벗어나지 않게 제한
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    cropped_face = image[y1:y2, x1:x2]
    return cropped_face, (x1, y1, x2, y2)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    # 지속적으로 얼굴 검출 관련 루프를 진행 (model_selection = 0, 모델은 근거리 검출 모델 사용, 신뢰도 50% 이상시 검출)
    print("얼굴 검출기 실행")
    # 왜 with문을 쓰는가? 리소스 정리까지 처리하기 때문
    while cap.isOpened(): #카메라가 열린동안
        success, frame = cap.read() #프레임을 읽으면 루프 지속, 아니면 루프종료
        # read가 튜플 형태의 값을 반환. - 이때 success는 boolean 값으로 값을 받아왔는지 여부, frame은 배열 형태
        if not success:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # OpenCV가 이미지를 BGR 순서로 읽는데, Mediapipe는 RGB 순서로 기대해서 바꿔줘야함 == COLOR_BGR2RGB
        results = face_detection.process(image) #이미지에서 얼굴 검출
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #OpenCV는 BGR만 제대로 출력하기에 BGR로 재조정

        if results.detections:
            for i, detection in enumerate(results.detections):
                face_crop, _ = crop_face_dynamic_size(image, detection)
                if face_crop is not None:
                    # 감정 분석 수행
                    emotion_scores = emotion_Analysis(face_crop)

                    if emotion_scores:
                        info_img = np.ones((200, 300, 3), dtype=np.uint8) * 255
                        for idx, (emotion, score) in enumerate(emotion_scores.items()):
                            text = f"{emotion}: {score:.2f}"
                            position = (10, 30 + idx * 25)
                            cv2.putText(info_img, text, position,
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.imshow('Emotion Probabilities', info_img)    
                        
                    # 얼굴만 출력하고 싶다면 아래 유지
                    cv2.imshow(f'Face Crop {i}', face_crop)
        
        cv2.imshow('Mediapipe Face Detection', image) #Cv2로 이미지 띄움 (해당 작업이 프레임 단위로 되므로 이미지가 영상처럼 보임)
        if cv2.waitKey(5) & 0xFF == 27: #Esc가 들어오면 종료
            break

cap.release() #열었던 웹캠을 닫는 함수
cv2.destroyAllWindows() # CV2로 띄운 모든 창을 닫는 함수


# In[ ]:




