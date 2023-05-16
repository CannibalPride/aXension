from __future__ import division
import os
import cv2
import dlib
import imutils
import numpy as np
from .eye import Eye
from .calibration import Calibration


class GazeTracking(object):
    """
    This class tracks the user's gaze.
    It provides useful information like the position of the eyes
    and pupils and allows to know if the eyes are open or closed
    """

    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()
        self.rectangle_shape = None

        self.image_points = None
        self.model_points = None

        self.b1 = None
        self.b2 = None
        self.b3 = None
        self.b4 = None
        self.b11 = None
        self.b12 = None
        self.b13 = None
        self.b14 = None

        # _face_detector is used to detect faces
        self._face_detector = dlib.get_frontal_face_detector()

        # _predictor is used to get facial landmarks of a given face
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "trained_models/shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)

    @staticmethod
    def draw_line(frame, a, b, color=(255, 255, 0)):
        cv2.line(frame, a, b, color, 10)

    @property
    def pupils_located(self):
        """Check that the pupils have been located"""
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    def _analyze(self):
        """Detects the face and initialize Eye objects"""
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector(frame, 0)
        size = frame.shape

        try:
            landmarks = self._predictor(frame, faces[0])
            self.eye_left = Eye(frame, landmarks, 0, self.calibration)
            self.eye_right = Eye(frame, landmarks, 1, self.calibration)

            self.image_points = np.array([
                (landmarks.part(33).x, landmarks.part(33).y),  # Nose tip
                (landmarks.part(8).x, landmarks.part(8).y),  # Chin
                (landmarks.part(36).x, landmarks.part(36).y),  # Left eye left corner
                (landmarks.part(45).x, landmarks.part(45).y),  # Right eye right corne
                (landmarks.part(48).x, landmarks.part(48).y),  # Left Mouth corner
                (landmarks.part(54).x, landmarks.part(54).y)  # Right mouth corner
            ], dtype="double")

            # 3D model points.
            self.model_points = np.array([
                (0.0, 0.0, 0.0),  # Nose tip
                (0.0, -330.0, -65.0),  # Chin
                (-225.0, 170.0, -135.0),  # Left eye left corner
                (225.0, 170.0, -135.0),  # Right eye right corner
                (-150.0, -150.0, -125.0),  # Left Mouth corner
                (150.0, -150.0, -125.0)  # Right mouth corner
            ])

            focal_length = size[1]
            center = (size[1] / 2, size[0] / 2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype="double"
            )

            dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
            (success, rotation_vector, translation_vector) = cv2.solvePnP(self.model_points, self.image_points, camera_matrix,
                                                                          dist_coeffs)

            (self.b1, jacobian) = cv2.projectPoints(np.array([(350.0, 270.0, 0.0)]), rotation_vector, translation_vector,
                                               camera_matrix, dist_coeffs)
            (self.b2, jacobian) = cv2.projectPoints(np.array([(-350.0, -270.0, 0.0)]), rotation_vector,
                                               translation_vector, camera_matrix, dist_coeffs)
            (self.b3, jacobian) = cv2.projectPoints(np.array([(-350.0, 270, 0.0)]), rotation_vector, translation_vector,
                                               camera_matrix, dist_coeffs)
            (self.b4, jacobian) = cv2.projectPoints(np.array([(350.0, -270.0, 0.0)]), rotation_vector,
                                               translation_vector, camera_matrix, dist_coeffs)

            (self.b11, jacobian) = cv2.projectPoints(np.array([(450.0, 350.0, 400.0)]), rotation_vector,
                                                translation_vector, camera_matrix, dist_coeffs)
            (self.b12, jacobian) = cv2.projectPoints(np.array([(-450.0, -350.0, 400.0)]), rotation_vector,
                                                translation_vector, camera_matrix, dist_coeffs)
            (self.b13, jacobian) = cv2.projectPoints(np.array([(-450.0, 350, 400.0)]), rotation_vector,
                                                translation_vector, camera_matrix, dist_coeffs)
            (self.b14, jacobian) = cv2.projectPoints(np.array([(450.0, -350.0, 400.0)]), rotation_vector,
                                                translation_vector, camera_matrix, dist_coeffs)

            self.b1 = (int(self.b1[0][0][0]), int(self.b1[0][0][1]))
            self.b2 = (int(self.b2[0][0][0]), int(self.b2[0][0][1]))
            self.b3 = (int(self.b3[0][0][0]), int(self.b3[0][0][1]))
            self.b4 = (int(self.b4[0][0][0]), int(self.b4[0][0][1]))

            self.b11 = (int(self.b11[0][0][0]), int(self.b11[0][0][1]))
            self.b12 = (int(self.b12[0][0][0]), int(self.b12[0][0][1]))
            self.b13 = (int(self.b13[0][0][0]), int(self.b13[0][0][1]))
            self.b14 = (int(self.b14[0][0][0]), int(self.b14[0][0][1]))

            newRect = dlib.rectangle(int(faces[0].left()), int(faces[0].top()), int(faces[0].right()), int(faces[0].bottom()))
            # Find face landmarks by providing rectangle for each face
            self.rectangle_shape = self._predictor(frame, newRect)

        except IndexError:
            self.eye_left = None
            self.eye_right = None

    def refresh(self, frame):
        """Refreshes the frame and analyzes it.

        Arguments:
            frame (numpy.ndarray): The frame to analyze
        """
        self.frame = frame
        self._analyze()

    def pupil_left_coords(self):
        """Returns the coordinates of the left pupil"""
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)

    def pupil_right_coords(self):
        """Returns the coordinates of the right pupil"""
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)

    def horizontal_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        horizontal direction of the gaze. The extreme right is 0.0,
        the center is 0.5 and the extreme left is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 10)
            pupil_right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def vertical_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        vertical direction of the gaze. The extreme top is 0.0,
        the center is 0.5 and the extreme bottom is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 10)
            pupil_right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def is_right(self):
        """Returns true if the user is looking to the right"""
        if self.pupils_located:
            return self.horizontal_ratio() <= 0.6

    def is_left(self):
        """Returns true if the user is looking to the left"""
        if self.pupils_located:
            return self.horizontal_ratio() >= 0.9

    def is_center(self):
        """Returns true if the user is looking to the center"""
        if self.pupils_located:
            return self.is_right() is not True and self.is_left() is not True

    def is_blinking(self):
        """Returns true if the user closes his eyes"""
        if self.pupils_located:
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
            return blinking_ratio > 3.8

    def annotated_frame(self):
        """Returns the main frame with pupils highlighted"""
        frame = self.frame.copy()

        if self.pupils_located:

            # Mark Pupils
            color = (0, 255, 0)
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)

            # Draw Landmarks
            for p in self.rectangle_shape.parts():
                cv2.circle(frame, (p.x, p.y), 2, (0, 255, 0), -1)

            self.draw_line(frame, self.b1, self.b3)
            self.draw_line(frame, self.b3, self.b2)
            self.draw_line(frame, self.b2, self.b4)
            self.draw_line(frame, self.b4, self.b1)

            self.draw_line(frame, self.b11, self.b13)
            self.draw_line(frame, self.b13, self.b12)
            self.draw_line(frame, self.b12, self.b14)
            self.draw_line(frame, self.b14, self.b11)

            self.draw_line(frame, self.b11, self.b1, color=(0, 0, 255))
            self.draw_line(frame, self.b13, self.b3, color=(0, 0, 255))
            self.draw_line(frame, self.b12, self.b2, color=(0, 0, 255))
            self.draw_line(frame, self.b14, self.b4, color=(0, 0, 255))

        return frame