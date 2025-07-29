from keras.models import load_model
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
import numpy as np

# 모델 로드
emotion_model_path = 'fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_classifier = load_model(emotion_model_path, compile=False)

# 감정 레이블
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# 감정 분석 함수
def emotion_Analysis(crop_image):
    try:
        predictions = emotion_classifier.predict(crop_image, verbose=0)[0]  # 확률 벡터
        
        emotion_scores = dict(zip(emotion_labels, predictions))
        return emotion_scores
    except Exception as e:
        print(f"[감정 분석 실패]: {e}")
        return None

def update_last_layer(user_images, user_labels, epochs=10, lr=1e-4):
    # 전 층 고정
    for layer in emotion_classifier.layers:
        layer.trainable = False
    emotion_classifier.layers[-1].trainable = True

    emotion_classifier.compile(optimizer=Adam(lr), loss=CategoricalCrossentropy())

    emotion_classifier.fit(np.array(user_images), np.array(user_labels), epochs=epochs, verbose=1)

    print("[모델 마지막 레이어 튜닝 완료]")