from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

MAX_POINTS = 1000
FRAME_INTERVAL = 5  # 30프레임마다 저장

# 감정 이력 저장 구조
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

def save_score(emotion_scores, frame_index):
    for emotion in emotion_history:
        emotion_history[emotion].append(emotion_scores.get(emotion, 0.0))
    time_history.append(frame_index)

def plot_emotion_history_to_cv2img(smoothing_window=5):
    if len(time_history) < 2:
        print(f"[부족한 데이터] 현재 time_history 길이: {len(time_history)}")
        return None

    def moving_average(data, window):
        if len(data) < window:
            return data  # 부족하면 그대로 반환
        return np.convolve(data, np.ones(window)/window, mode='valid')

    fig, ax = plt.subplots(figsize=(8, 4))

    times = [f / 30 for f in time_history]
    min_len = min([len(scores) for scores in emotion_history.values()])

    for emotion, scores in emotion_history.items():
        smoothed_scores = moving_average(list(scores), smoothing_window)
        # 시간축도 맞춰줌
        trimmed_times = times[-len(smoothed_scores):]
        ax.plot(trimmed_times, smoothed_scores, label=emotion)

    ax.set_xlabel('Time (3Frame)')
    ax.set_ylabel('Emotion Score')
    ax.set_title('Emotion Trends (Smoothed)')
    ax.legend(loc='upper right')
    fig.tight_layout()

    canvas = FigureCanvas(fig)
    canvas.draw()

    img = np.asarray(canvas.buffer_rgba())
    img_cv2 = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    plt.close(fig)
    return img_cv2

