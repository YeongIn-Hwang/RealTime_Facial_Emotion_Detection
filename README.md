RealTime Facial Emotion Detection
https://github.com/YeongIn-Hwang/RealTime_Facial_Emotion_Detection

실시간 얼굴 검출 및 감정 분석 시스템 (Mini-Xception 기반)
- 얼굴을 실시간으로 인식하고, 감정을 분석하여 시각적으로 보여주는 Python 기반 데스크탑 애플리케이션
- Media pipe로 얼굴을 검출하고, Mini-Xception 모델을 통해 7가지 감정을 분류

Ｎｏｔｉｏｎ 링크
https://www.notion.so/22ab2a809d2280df8e28fb8377085b12

모델 출처
https://github.com/oarriaga/face_classification
=================================================================
Features
- 실시간 얼굴 검출: Mediapipe 사용, 눈·코·입 기반 얼굴 위치 동적 추출
- 감정 인식 모델: Mini-Xception 기반 감정 분류 (7가지 감정)
- 시각화 UI
	- 실시간 감정 확률 패널 (OpenCV 기반)
	- 감정 시간 흐름 그래프 (Matplotlib 기반, OpenCV 연동)
- 이력 저장: 최대 1000개 프레임까지 감정 확률 시계열 저장

==================================================================
Architecture
웹캠 영상
  ↓
Mediapipe 얼굴 검출
  ↓
눈-코-입 중심으로 얼굴 crop
  ↓
Grayscale + Resize + 정규화 전처리
  ↓
Mini-Xception 모델 입력
  ↓
7개 감정 확률 예측
  ↓
1) 실시간 감정 텍스트 형태로 출력 (cv2.putText)
2) 감정 추세 그래프 시각화 (Matplotlib + OpenCV)
===================================================================
How to Run
Face_Detection_Main.py 를 python으로 실행
ESC 누르면 종료

===================================================================
File Structure
.
├── Face_Detection_Main.py       # 실시간 얼굴 검출 + 감정 분석 메인
├── Emotion_Analysis.py            # Mini-Xception 모델 로딩 및 분석
├── Preprocessing.py               # 입력 이미지 전처리 (grayscale, resize 등)
├── Visualize.py                   # 감정 추세 그래프 시각화 모듈
├── fer2013_mini_XCEPTION.102-0.66.hdf5     # 사전 학습된 감정 분류 모델
├── requirements.txt
└── README.md

===================================================================
Author
YeongIn-Hwang