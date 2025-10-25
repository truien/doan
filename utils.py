import os
from datetime import datetime

import cv2
import numpy as np
import pandas as pd


def draw_line_segment(frame, p1, p2):
    cv2.line(frame, p1, p2, (0, 0, 255), 3)


def preprocess_frame(frame):
    return cv2.GaussianBlur(frame, (5, 5), 0)


def save_count_stats(video_name, count_in, count_out):
    csv_path = "results/summary.csv"
    os.makedirs("results", exist_ok=True)

    if not os.path.exists(csv_path):
        df = pd.DataFrame(
            columns=["video_name", "count_in", "count_out", "total", "datetime"])
    else:
        df = pd.read_csv(csv_path)

    new_row = {
        "video_name": video_name,
        "count_in": count_in,
        "count_out": count_out,
        "total": count_in + count_out,
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(csv_path, index=False)


def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame
