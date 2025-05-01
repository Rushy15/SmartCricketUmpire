import cv2
import numpy as np

DEVIATION_THRESHOLD = 100

class BallKalmanTracker:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)  # 4 states, 2 measurements (x, y)
        
        # Transition matrix (A)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],  # x' = x + vx
            [0, 1, 0, 1],  # y' = y + vy
            [0, 0, 1, 0],  # vx' = vx
            [0, 0, 0, 1]   # vy' = vy
        ], dtype=np.float32)

        self.kalman.measurementMatrix = np.eye(2, 4, dtype=np.float32) # Measurement matrix (H)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2 # Process noise covariance (Q)
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1 # Measurement noise covariance (R)

        self.path = [] # list to store empty trajectories
        self.centres = []
        self.impact_point = None
        self.last_prediction = None

    def predict(self):
        prediction = self.kalman.predict()
        self.last_prediction = (int(prediction[0]), int(prediction[1]))
        return self.last_prediction

    def correct(self, centre):
        measured = np.array([[np.float32(centre[0])], [np.float32(centre[1])]])
        self.kalman.correct(measured)

    def track_ball(self, centre):
        """
        Update Kalman Filter with a new measurement (detected ball centre).
        """
        
        self.predict()
        self.correct(centre)
        self.centres.append(centre)
        
        if self.impact_point == None:
            self.path.append(centre)

            if len(self.path) >= 2:
                delta_x = abs(self.path[-1][0] - self.path[-2][0]) # calculating the difference in x-drection between the two most recent entries in ball trajectories
                if delta_x > DEVIATION_THRESHOLD:
                    self.impact_point = self.path[-2]

    def predict_without_detection(self):
        self.predict()

        if self.impact_point == None:
            self.path.append(self.last_prediction)

    def draw_path(self, frame):
        if len(self.path) >= 2 and self.impact_point == None:
            for i in range(1, len(self.path)):
                cv2.line(frame, self.path[i - 1], self.path[i], (255, 0, 0), 3)

    def mark_impact_point(self, frame):
        if self.impact_point:
            x, y = self.impact_point
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(frame, '*', (x - 15, y + 15), font, 2, (0, 0 , 255), 3)
    
    def get_impact_point(self):
        return self.impact_point