from fer.fer import FER  # 이 방식이면 moviepy 문제 회피 가능
import cv2

# 전역으로 detector 한 번만 초기화
emotion_detector = FER(mtcnn=True)

def emotion_Analysis(crop_image):
    try:
        result = emotion_detector.detect_emotions(crop_image)
        if result and 'emotions' in result[0]:
            return result[0]['emotions']  # 전체 감정 확률 딕셔너리 반환
        else:
            return None
    except:
        return None