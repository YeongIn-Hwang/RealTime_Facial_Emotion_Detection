from keras.models import load_model
import cv2
import numpy as np

# 모델 로드
emotion_model_path = 'fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_classifier = load_model(emotion_model_path, compile=False)

# 감정 레이블
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# 감정 분석 함수
def emotion_Analysis(crop_image):
    try:
        #gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
        #resized_face = cv2.resize(gray, (64, 64)) / 255.0
        #input_face = np.expand_dims(resized_face, axis=(0, -1))  # (1, 64, 64, 1)
        predictions = emotion_classifier.predict(crop_image, verbose=0)[0]  # 확률 벡터
        
        emotion_scores = dict(zip(emotion_labels, predictions))
        return emotion_scores
    except Exception as e:
        print(f"[감정 분석 실패]: {e}")
        return None
