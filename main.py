import sys
import cv2
import numpy as np
import sqlite3
import time
from collections import Counter
from datetime import datetime
from deepface import DeepFace
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer
import matplotlib.pyplot as plt

DB_PATH = "emotions.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS emotions (
        id INTEGER PRIMARY KEY,
        ts TEXT,
        date TEXT,
        emotion TEXT
    )
    """)
    conn.commit()
    conn.close()

def log_emotion(emotion):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now()
    c.execute("INSERT INTO emotions (ts, date, emotion) VALUES (?,?,?)",
              (now.isoformat(), now.date().isoformat(), emotion))
    conn.commit()
    conn.close()

def get_summary():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT date, emotion, COUNT(*) FROM emotions GROUP BY date, emotion ORDER BY date")
    rows = c.fetchall()
    conn.close()
    return rows

def show_summary():
    rows = get_summary()
    if not rows:
        print("No data logged yet.")
        return

    summary = {}
    for d, emo, cnt in rows:
        summary.setdefault(d, {})
        summary[d][emo] = cnt

    print("\n=== Daily Emotion Summary ===")
    for d, emos in summary.items():
        print(f"{d}: {emos}")

    dates = list(summary.keys())
    emotions = sorted({e for _, emos in summary.items() for e in emos.keys()})
    counts_by_emotion = {emo: [summary[d].get(emo, 0) for d in dates] for emo in emotions}

    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = [0] * len(dates)
    for emo in emotions:
        ax.bar(dates, counts_by_emotion[emo], bottom=bottom,
               label=emo, color=np.array(emotion_colors.get(emo, (128,128,128)))/255)
        bottom = [b + c for b, c in zip(bottom, counts_by_emotion[emo])]

    ax.set_xlabel("Date")
    ax.set_ylabel("Emotion Count")
    ax.set_title("Daily Emotion Trends")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

emotion_colors = {
    "happy": (0, 255, 0),
    "neutral": (200, 200, 200),
    "sad": (255, 0, 0),
    "angry": (0, 0, 255),
    "surprise": (255, 255, 0),
    "fear": (128, 0, 128),
    "disgust": (0, 128, 0)
}

feedback_tips = {
    "happy": "Smile back and engage positively!",
    "neutral": "Maintain calm and attentive posture.",
    "sad": "Offer support or ask 'Are you okay?'",
    "angry": "Stay calm and listen carefully.",
    "surprise": "React with curiosity and interest.",
    "fear": "Reassure them and create a safe space.",
    "disgust": "Respect their reaction and adjust behavior."
}

class NeuroLensApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroLens - DeepFace Emotion Feedback")
        self.setGeometry(100, 100, 960, 720)

        self.layout = QVBoxLayout()
        self.camera_label = QLabel("Camera feed will appear here")
        self.camera_label.setFont(QFont("Arial", 12))
        self.feedback_label = QLabel("Feedback will appear here")
        self.feedback_label.setFont(QFont("Arial", 14))
        self.feedback_label.setStyleSheet("color: blue;")
        self.layout.addWidget(self.camera_label)
        self.layout.addWidget(self.feedback_label)

        self.summary_btn = QPushButton("Show Summary")
        self.summary_btn.clicked.connect(show_summary)
        self.layout.addWidget(self.summary_btn)

        self.quit_btn = QPushButton("Quit")
        self.quit_btn.clicked.connect(self.close)
        self.layout.addWidget(self.quit_btn)
        self.setLayout(self.layout)

        self.cap = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.face_states = {}
        self.frame_count = 0
        self.update_interval = 30  # ~1 sec per emotion update at 30 FPS
        self.prev_color_default = np.array([200,200,200], dtype=np.float32)

        # logging control
        self.last_log_time = 0
        self.LOG_INTERVAL = 10 

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)



        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)


    def detect_emotion(self, face_rgb):
        try:
            result = DeepFace.analyze(face_rgb, actions=['emotion'], enforce_detection=False)
            return result[0]['dominant_emotion']
        except:
            return "neutral"

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        self.frame_count += 1
        new_face_states = {}

        for i, (x, y, w, h) in enumerate(faces):
            x, y, w, h = map(int, (x, y, w, h))
            face_rgb = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)

            prev = self.face_states.get(i, ["neutral", self.prev_color_default])
            emotion, prev_color = prev

            if self.frame_count % self.update_interval == 0:
                emotion = self.detect_emotion(face_rgb)

            new_color = np.array(emotion_colors.get(emotion, (255,255,255)), dtype=np.float32)
            prev_color = prev_color*0.8 + new_color*0.2
            overlay_color = (int(prev_color[0]), int(prev_color[1]), int(prev_color[2]))

            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x+w, y+h), overlay_color, -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, overlay_color, 2)

            new_face_states[i] = [emotion, prev_color]

        self.face_states = new_face_states

        if new_face_states:
            emotion_list = [v[0] for v in new_face_states.values()]
            dominant_emotion = Counter(emotion_list).most_common(1)[0][0]
        else:
            dominant_emotion = "neutral"

        tip = feedback_tips.get(dominant_emotion, "")
        self.feedback_label.setText(f"Mirror Feedback: {dominant_emotion.upper()} | Tip: {tip}")

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        self.cap.release()
        cv2.destroyAllWindows()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NeuroLensApp()
    window.show()
    sys.exit(app.exec_())
