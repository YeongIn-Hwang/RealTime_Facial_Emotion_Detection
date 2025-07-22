{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "58d47dbc",
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2\n",
    "import mediapipe as mp\n",
    "\n",
    "mp_face_detection = mp.solutions.face_detection # 얼굴 검출 모델 준비\n",
    "mp_drawing = mp.solutions.drawing_utils # 얼굴 검출된 부분에 사각형 유틸 추가\n",
    "\n",
    "cap = cv2.VideoCapture(0) #0번 장치로 기본 카메라를 오픈\n",
    "\n",
    "with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:\n",
    "    # 지속적으로 얼굴 검출 관련 루프를 진행 (model_selection = 0, 모델은 근거리 검출 모델 사용, 신뢰도 50% 이상시 검출)\n",
    "    # 왜 with문을 쓰는가? 리소스 정리까지 처리하기 때문\n",
    "    while cap.isOpened(): #카메라가 열린동안\n",
    "        success, frame = cap.read() #프레임을 읽으면 루프 지속, 아니면 루프종료\n",
    "        # read가 튜플 형태의 값을 반환. - 이때 success는 boolean 값으로 값을 받아왔는지 여부, frame은 배열 형태\n",
    "        if not success:\n",
    "            break\n",
    "\n",
    "        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\n",
    "        # OpenCV가 이미지를 BGR 순서로 읽는데, Mediapipe는 RGB 순서로 기대해서 바꿔줘야함 == COLOR_BGR2RGB\n",
    "        results = face_detection.process(image) #이미지에서 얼굴 검출\n",
    "        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #OpenCV는 BGR만 제대로 출력하기에 BGR로 재조정\n",
    "\n",
    "        if results.detections:\n",
    "            for detection in results.detections:\n",
    "                mp_drawing.draw_detection(image, detection) # 얼굴 검출된게 있으면 시각화(눈,코,입 등 출력)\n",
    "\n",
    "        cv2.imshow('Mediapipe Face Detection', image) #Cv2로 이미지 띄움 (해당 작업이 프레임 단위로 되므로 이미지가 영상처럼 보임)\n",
    "        if cv2.waitKey(5) & 0xFF == 27: #Esc가 들어오면 종료\n",
    "            break\n",
    "\n",
    "cap.release() #열었던 웹캠을 닫는 함수\n",
    "cv2.destroyAllWindows() # CV2로 띄운 모든 창을 닫는 함수\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "fer-env",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.18"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
