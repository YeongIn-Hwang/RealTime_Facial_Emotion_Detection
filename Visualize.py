from collections import deque
import matplotlib.pyplot as plt
import time
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

MAX_POINTS = 1000
emotion_history = {
    'happy': deque(maxlen=MAX_POINTS),
    'sad': deque(maxlen=MAX_POINTS),
    'angry': deque(maxlen=MAX_POINTS),
    'fear': deque(maxlen=MAX_POINTS),
    'surprise': deque(maxlen=MAX_POINTS),
    'neutral': deque(maxlen=MAX_POINTS),
    'disgust': deque(maxlen=MAX_POINTS)
}
time_history = deque(maxlen=MAX_POINTS)

def save_score(emotion_scores):
    current_time = time.time()
    time_history.append(current_time)
    for emotion in emotion_history:
        emotion_history[emotion].append(emotion_scores.get(emotion, 0.0))

def plot_emotion_history_to_cv2img():
    if len(time_history) < 2:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    for emotion, scores in emotion_history.items():
        ax.plot(list(time_history), list(scores), label=emotion)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Emotion Score')
    ax.set_title('Emotion Trends')
    ax.legend(loc='upper right')
    fig.tight_layout()

    # matplotlib figure → image (PIL → numpy → cv2)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    img_pil = Image.open(buf).convert("RGB")
    img_np = np.array(img_pil)
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    return img_cv2

