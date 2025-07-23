from fer import FER
import cv2

def emotion_Analysis(crop_image):
    emotion_detector = FER(mtcnn=True)  # or mtcnn=False if slow
    result = emotion_detector.detect_emotions(crop_image)
    return result

