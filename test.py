import cv2
import numpy as np

# Load the pre-trained Haar cascade for detecting eyes
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Capture video from the default camera
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect eyes in the frame
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangles around the detected eyes
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Calculate the center of each eye
        eye_center_x = ex + ew // 2
        eye_center_y = ey + eh // 2

        # Determine the quadrant based on the eye center coordinates
        height, width, _ = frame.shape
        if eye_center_x < width // 2:
            if eye_center_y < height // 2:
                quadrant = "Top Left"
            else:
                quadrant = "Bottom Left"
        else:
            if eye_center_y < height // 2:
                quadrant = "Top Right"
            else:
                quadrant = "Bottom Right"

        # Display the quadrant on the frame
        cv2.putText(frame, quadrant, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Eye Tracking', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
