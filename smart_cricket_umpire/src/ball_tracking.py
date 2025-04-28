# ball_tracking.py
from collections import deque
import cv2

class BallTracker:
    def __init__(self, deviation_threshold=30):
        self.ball_positions = deque(maxlen=30)
        self.tracking_active = True
        self.impact_point = None
        self.deviation_threshold = deviation_threshold

    def track_ball(self, ball_box):
        if ball_box and self.tracking_active:
            if len(self.ball_positions) > 0:
                prev_x, _ = self.ball_positions[-1]
                curr_x, _ = ball_box
                if abs(curr_x - prev_x) > self.deviation_threshold:
                    self.tracking_active = False
                    self.impact_point = self.ball_positions[-1]  # Save impact point before deviation
                else:
                    self.ball_positions.append(ball_box)
            else:
                self.ball_positions.append(ball_box)
        print("Ball Box is: ", ball_box) # De-bugging

    def draw_path(self, frame):
        for i in range(1, len(self.ball_positions)):
            cv2.line(frame, self.ball_positions[i - 1], self.ball_positions[i], (255, 0, 0), 2)

    def mark_impact_point(self, frame):
        if self.impact_point:
            cv2.putText(frame, "*", (self.impact_point[0] - 5, self.impact_point[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    def get_impact_point(self):
        return self.impact_point
