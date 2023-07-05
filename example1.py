import cv2
from gaze_tracking import GazeTracking
import imutils
import mediapipe as mp
from gaze_tracking import gaze as gz

import time

mp_face_mesh = mp.solutions.face_mesh  # initialize the face mesh model
text = "Not Found"

if __name__ == "__main__":
    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,  # number of faces to track in each frame
            refine_landmarks=True,  # includes iris landmarks in the face mesh model
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as face_mesh:
        while True:
            # We get a new frame from the webcam
            success, frame = webcam.read()
            if not success:  # no frame input
                print(text)

            frame = imutils.resize(frame, width=1600)
            frame.flags.writeable = False

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # frame to RGB for the face-mesh model
            results = face_mesh.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # frame back to BGR for OpenCV
            if results.multi_face_landmarks:
                gz.gaze(frame, results.multi_face_landmarks[0])  # gaze estimation

            # We send this frame to GazeTracking to analyze it
            gaze.refresh(frame)

            frame = gaze.annotated_frame()

            """
            if gaze.is_blinking():
                text = "Blinking"
            elif gaze.is_right():
                text = "Tracking: Looking right"
            elif gaze.is_left():
                text = "Tracking: Looking left"
            elif gaze.is_center():
                text = "Tracking: Looking center"

            cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_COMPLEX, 0.8, (147, 31, 58), 2)
            """

            #cv2.putText(frame, "Left pupil:  " + str(gaze.pupil_left_coords()), (90, 90), cv2.FONT_HERSHEY_COMPLEX, 0.5, (147, 31, 58), 1)
            #cv2.putText(frame, "Right pupil: " + str(gaze.pupil_right_coords()), (90, 110), cv2.FONT_HERSHEY_COMPLEX, 0.5, (147, 31, 58), 1)
            #cv2.putText(frame, "Vertical Ratio:  " + str(gaze.horizontal_ratio()), (90, 125), cv2.FONT_HERSHEY_COMPLEX, 0.5, (147, 31, 58), 1)
            #cv2.putText(frame, "Horizontal Ratio: " + str(gaze.vertical_ratio()), (90, 145), cv2.FONT_HERSHEY_COMPLEX, 0.5, (147, 31, 58), 1)

            # Display the log box
            queue_text = "Log: "
            recent_anomaly = ""
            cv2.putText(frame, queue_text, (120, 60), cv2.FONT_HERSHEY_COMPLEX, 0.8, (147, 31, 58), 2)
            if not gaze.anomaly_queue_log.empty():
                recent_anomaly = f"{time.time()} - {gaze.anomaly_queue_log.get()}"

            cv2.putText(frame, recent_anomaly, (120, 90), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 1)

            cv2.imshow("PyGaze", frame)

            if cv2.waitKey(1) == 27:
                break

    webcam.release()
    cv2.destroyAllWindows()
