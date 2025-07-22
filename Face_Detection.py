#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection # 얼굴 검출 모델 준비
mp_drawing = mp.solutions.drawing_utils # 얼굴 검출된 부분에 사각형 유틸 추가

cap = cv2.VideoCapture(0) #0번 장치로 기본 카메라를 오픈

if not (cap.isOpened()):
    print("웹캠이 열리지 않습니다.")
    exit()

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
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection) # 얼굴 검출된게 있으면 시각화(눈,코,입 등 출력)

        cv2.imshow('Mediapipe Face Detection', image) #Cv2로 이미지 띄움 (해당 작업이 프레임 단위로 되므로 이미지가 영상처럼 보임)
        if cv2.waitKey(5) & 0xFF == 27: #Esc가 들어오면 종료
            break

cap.release() #열었던 웹캠을 닫는 함수
cv2.destroyAllWindows() # CV2로 띄운 모든 창을 닫는 함수


# %%
