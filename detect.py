import csv
import os
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from boxmot.trackers.botsort.botsort import BotSort
from ultralytics import YOLO
from utils import draw_line_segment, save_count_stats

YOLO_PATH = "model/yolo_trained.pt"
BOTSORT_PATH = "model/osnet_x0_25_msmt17.pt"

COOLDOWN_FRAMES = 1000
CENTER_THRESHOLD = 100


def log_event(video_name, track_id, direction):
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", "detailed_logs.csv")
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            video_name,
            track_id,
            direction
        ])


def side_of_line(p, a, b):
    val = (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
    if abs(val) < 1e-3:
        return 0
    return np.sign(val)


def distance_point_to_line(p, a, b):
    """Khoảng cách từ p đến đường thẳng AB"""
    x0, y0 = p
    x1, y1 = a
    x2, y2 = b
    return abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)


def point_near_segment(p, a, b, max_dist=10):
    """Kiểm tra xem điểm p có gần đoạn AB không"""
    a, b, p = np.array(a), np.array(b), np.array(p)
    ab = b - a
    ap = p - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    if t < 0 or t > 1:
        return False
    proj = a + t * ab
    dist = np.linalg.norm(p - proj)
    return dist < max_dist


def segments_intersect(p1, p2, a, b):
    """Kiểm tra xem đoạn (p1,p2) có cắt đoạn (a,b) không"""
    def ccw(pA, pB, pC):
        return (pC[1]-pA[1])*(pB[0]-pA[0]) > (pB[1]-pA[1])*(pC[0]-pA[0])
    return ccw(p1, a, b) != ccw(p2, a, b) and ccw(p1, p2, a) != ccw(p1, p2, b)


def process_frame_stream(video_path, line_points, direction_points):
    """Đếm người qua line theo tâm + hiển thị vùng nhận diện"""
    # an
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return

    try:
        # yolo có hỗ trợ mps không

        model = YOLO(YOLO_PATH)
        tracker = BotSort(
            reid_weights=Path(BOTSORT_PATH),
            # dùng apple mps nếu có
            device="mps",
            max_age=9999,
            track_buffer=9999,
            half=False,
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Cannot open video")
            return

        count_in, count_out = 0, 0
        prev_center = {}
        last_count_frame = {}
        frame_count = 0
        video_name = os.path.basename(video_path)

        a, b = line_points
        ref_in, ref_out = direction_points
        reference_side = side_of_line(ref_out, a, b)

        while True:
            ret, frame = cap.read()
            if not ret:
                save_count_stats(video_name, count_in, count_out)
                break

            frame_count += 1
            try:
                results = model.predict(
                    frame, conf=0.5, verbose=False)
                boxes = results[0].boxes
                if boxes is None or len(boxes) == 0:
                    continue

                dets = boxes.data.cpu().numpy()
                dets = dets[dets[:, 5] == 0]
                tracks = tracker.update(dets, frame)

                draw_line_segment(frame, *line_points)
                cv2.line(frame, ref_out, ref_in, (255, 0, 255), 2)
                cv2.putText(frame, "OUT", ref_out,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, "IN", ref_in,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                for x1, y1, x2, y2, tid, conf, cls, _ in tracks.astype(float):
                    tid = int(tid)
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    center = (cx, cy)

                    # --- Kiểm tra vị trí ---
                    curr_side = side_of_line(center, a, b)
                    crossed = False

                    if tid in prev_center:
                        if segments_intersect(prev_center[tid], center, a, b):
                            crossed = True
                    prev_center[tid] = center

                    near_line = point_near_segment(
                        center, a, b, max_dist=CENTER_THRESHOLD)

                    color = (255, 255, 0)
                    if near_line:
                        color = (0, 255, 255)

                    cv2.circle(frame, (int(cx), int(cy)),
                               CENTER_THRESHOLD, color, -1)
                    cv2.putText(frame, f'ID: {tid}', (int(x1), int(y1)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    if crossed and near_line:
                        last_f = last_count_frame.get(tid, -9999)
                        if frame_count - last_f > COOLDOWN_FRAMES:
                            if curr_side == reference_side:
                                count_in += 1
                                log_event(video_name, tid, "IN")
                                cv2.circle(frame, (int(cx), int(cy)),
                                           8, (0, 255, 0), -1)
                            else:
                                count_out += 1
                                log_event(video_name, tid, "OUT")
                                cv2.circle(frame, (int(cx), int(cy)),
                                           8, (0, 0, 255), -1)
                            last_count_frame[tid] = frame_count

            except Exception as e:
                print(f"⚠️ Detection error: {e}")
                continue

            cv2.putText(frame, f"In: {count_in} | Out: {count_out}",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            try:
                ret, buffer = cv2.imencode(".jpg", frame)
                if not ret:
                    continue
                frame_bytes = buffer.tobytes()
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            except Exception as e:
                print(f"❌ Frame encode error: {e}")

        cap.release()

    except Exception as e:
        print(f"❌ Error in process_frame_stream: {e}")
