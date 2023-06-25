"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking
import imutils

if __name__ == "__main__":
    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)

    while True:
        # We get a new frame from the webcam
        _, frame = webcam.read()
        frame = imutils.resize(frame, width=900)

        # We send this frame to GazeTracking to analyze it
        gaze.refresh(frame)

        frame = gaze.annotated_frame()
        text = "Not Found"
        #text = f"Tracking: {gaze.determine_gaze_direction()}"
        if gaze.is_blinking():
            text = "Blinking"
        elif gaze.is_right():
            text = "Tracking: Looking right"
        elif gaze.is_left():
            text = "Tracking: Looking left"
        elif gaze.is_center():
            text = "Tracking: Looking center"

        cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_COMPLEX, 0.8, (147, 31, 58), 2)

        cv2.putText(frame, "Left pupil:  " + str(gaze.pupil_left_coords()), (90, 90), cv2.FONT_HERSHEY_COMPLEX, 0.5, (147, 31, 58), 1)
        cv2.putText(frame, "Right pupil: " + str(gaze.pupil_right_coords()), (90, 110), cv2.FONT_HERSHEY_COMPLEX, 0.5, (147, 31, 58), 1)
        cv2.putText(frame, "Vertical Ratio:  " + str(gaze.horizontal_ratio()), (90, 125), cv2.FONT_HERSHEY_COMPLEX, 0.5, (147, 31, 58), 1)
        cv2.putText(frame, "Horizontal Ration: " + str(gaze.vertical_ratio()), (90, 145), cv2.FONT_HERSHEY_COMPLEX, 0.5, (147, 31, 58), 1)

        cv2.imshow("PyGaze", frame)

        if cv2.waitKey(1) == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()
