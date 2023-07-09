import os
import pprint

from flask import Flask, render_template, Response
import cv2
import imutils
from gaze_tracking import GazeTracking
import mediapipe as mp
import time

app = Flask(__name__)
gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    global webcam, gaze

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        while True:
            success, frame = webcam.read()
            if not success:
                break

            frame = imutils.resize(frame, width=1600)
            frame.flags.writeable = False

            gaze.refresh(frame)
            frame = gaze.annotated_frame()

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    if not os.path.exists('logs'):
        os.makedirs('logs')

    with open(f'logs/log.txt', 'w') as file:
        while not gaze.anomaly_queue_log.empty():
            item = pprint.pformat(gaze.anomaly_queue_log2.get())
            log_entry = f"{time.ctime(time.time())}: \n{item}\n\n"
            file.write(log_entry)

    app.run(debug=True, port=8080)

webcam.release()
cv2.destroyAllWindows()
