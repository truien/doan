import os
import re

import cv2
import pandas as pd
from flask import (Flask, Response, jsonify, redirect, render_template,
                   request, url_for)

from detect import process_frame_stream
from utils import get_first_frame

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

video_filename = None
line_points = None
try:
    direction_points
except NameError:
    direction_points = None


@app.route('/')
def index():
    videos = []
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        videos = [f for f in os.listdir(app.config['UPLOAD_FOLDER'])
                  if f.endswith('.mp4')]
    return render_template('index.html', videos=videos)


@app.route('/delete_video', methods=['POST'])
def delete_video():
    global video_filename, line_points, direction_points
    filename = request.form.get('filename', '')

    if not filename or not filename.endswith('.mp4'):
        return redirect(url_for('index'))

    safe_name = os.path.basename(filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)

    try:
        uploads_abs = os.path.abspath(app.config['UPLOAD_FOLDER'])
        target_abs = os.path.abspath(file_path)
        if os.path.commonpath([uploads_abs, target_abs]) != uploads_abs:
            return redirect(url_for('index'))

        if os.path.exists(file_path):
            os.remove(file_path)

            if video_filename == safe_name:
                video_filename = None
                line_points = None
                direction_points = None
    except Exception:
        pass

    return redirect(url_for('index'))


@app.route('/upload', methods=['POST'])
def upload_video():
    global video_filename
    file = request.files['video']

    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    if not file.filename.endswith('.mp4'):
        return jsonify({'error': 'Only .mp4 files are allowed'}), 400

    try:
        clean_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', file.filename)
        video_filename = clean_name

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        file.save(file_path)

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            os.remove(file_path)
            return jsonify({'error': 'Invalid video file'}), 400
        cap.release()

        return redirect(url_for('draw_line'))

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/select_video/<filename>')
def select_video(filename):
    global video_filename
    if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], filename)):
        video_filename = filename
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        frame = get_first_frame(video_path)
        import base64
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return render_template('draw_line.html',
                               filename=video_filename,
                               frame_data=frame_base64)
    return redirect(url_for('index'))


@app.route('/draw_line')
def draw_line():
    if not video_filename:
        return redirect(url_for('index'))

    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
    frame = get_first_frame(video_path)

    if frame is None:
        return jsonify({'error': 'Could not read video frame'}), 400

    _, buffer = cv2.imencode('.jpg', frame)
    frame_bytes = buffer.tobytes()

    return render_template('draw_line.html',
                           filename=video_filename,
                           frame_bytes=frame_bytes)


@app.route('/set_line', methods=['POST'])
def set_line():
    global line_points, direction_points
    try:
        x1 = int(request.form['x1'])
        y1 = int(request.form['y1'])
        x2 = int(request.form['x2'])
        y2 = int(request.form['y2'])
        line_points = ((x1, y1), (x2, y2))
        x3 = int(request.form['x3'])
        y3 = int(request.form['y3'])
        x4 = int(request.form['x4'])
        y4 = int(request.form['y4'])
        direction_points = ((x3, y3), (x4, y4))
        return redirect(url_for('detect'))
    except:
        return jsonify({'error': 'Invalid line coordinates'}), 400


@app.route('/detect')
def detect():
    if not video_filename or not line_points:
        return redirect(url_for('index'))
    return render_template('detect.html', filename=video_filename)


@app.route('/video_feed')
def video_feed():
    if not video_filename or not line_points or not direction_points:
        return Response(status=404)

    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
    return Response(
        process_frame_stream(video_path, line_points, direction_points),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/stats')
def stats():
    detailed_logs = []
    csv_path = "logs/detailed_logs.csv"

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, names=[
                         'datetime', 'video_name', 'track_id', 'direction'])
        detailed_logs = df.to_dict('records')

        # Calculate summary statistics
        summary_stats = []
        for video in df['video_name'].unique():
            video_data = df[df['video_name'] == video]
            count_in = len(video_data[video_data['direction'] == 'IN'])
            count_out = len(video_data[video_data['direction'] == 'OUT'])
            total = count_in + count_out
            latest_time = video_data['datetime'].max()

            summary_stats.append({
                'video_name': video,
                'count_in': count_in,
                'count_out': count_out,
                'total': total,
                'datetime': latest_time
            })

    return render_template('stats.html',
                           detailed_logs=detailed_logs,
                           summary_stats=summary_stats)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
