import cv2
import numpy as np

# Screen dimensions (adjust according to your screen resolution)
screen_width, screen_height = 1920, 1080

# Create an empty screen
screen = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load the pre-trained face and eye cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face and eye detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) within the face for eye detection
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes in the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            # Calculate the gaze position on the screen
            gaze_x = int((x + ex + ew / 2) * (screen_width / frame.shape[1]))
            gaze_y = int((y + ey + eh / 2) * (screen_height / frame.shape[0]))

            # Draw a marker at the estimated gaze position on the screen
            cv2.drawMarker(screen, (gaze_x, gaze_y), (0, 0, 255), cv2.MARKER_CROSS, markerSize=20, thickness=2)

    # Show the screen with the gaze marker
    cv2.imshow('Gaze Tracking', screen)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
