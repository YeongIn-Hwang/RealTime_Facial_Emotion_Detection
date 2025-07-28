import cv2
import numpy as np

def preprocess_input_for_miniXception(image_bgr):
    """
    MiniXception 모델용 전처리 함수.
    
    Args:
        image_bgr (np.ndarray): BGR 형식의 얼굴 이미지 (crop된 상태).
        
    Returns:
        np.ndarray: (1, 48, 48, 1) 형식의 전처리된 입력.
    """
    if image_bgr is None or image_bgr.size == 0:
        return None

    # 1. Grayscale 변환
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # 2. 48x48 크기로 resize
    resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)

    # 3. 정규화
    normalized = resized.astype("float32") / 255.0

    # 4. shape 맞추기 (48, 48, 1) → (1, 48, 48, 1)
    reshaped = np.expand_dims(normalized, axis=-1)  # 채널 축
    final_input = np.expand_dims(reshaped, axis=0)  # 배치 축

    return final_input