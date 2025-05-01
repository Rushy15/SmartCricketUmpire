# main.py
import cv2
from src.ball_tracking import BallTracker
from src.kalman_balltracker import BallKalmanTracker
from src.stumps_detection import detect_and_update_stumps, refine_stumps_with_edges
from src.utils import check_lbw
from ultralytics import YOLO
#=========================== SECTION TO CHANGE =====================================#
MODEL_NAME = "model2_best.pt"
INPUT_VIDEO_NAME = "lbw_test.mp4"
OUTPUT_VIDEO_NAME = "lbw_test_result.mp4"

IDENTIFICATION_CONFIDENCE = 0.3
#===================================================================================#
# Initialize the ball tracker and the model
ball_tracker = BallKalmanTracker()
MODEL_PATH = "models/" + str(MODEL_NAME) 
VIDEO_PATH = "input/testing_videos/" + str(INPUT_VIDEO_NAME)

CURRENT_STUMPS_BBOX = None

# Open video and create output
model = YOLO(MODEL_PATH)
vid = cv2.VideoCapture(VIDEO_PATH)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_NAME, fourcc, 30.0, (int(vid.get(3)), int(vid.get(4))))

while vid.isOpened():
    ret, frame = vid.read()
    if not ret:
        break
    
    # # Preprocessing: Convert to grayscale, apply Gaussian blur
    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # # Edge detection (optional)
    # edges = cv2.Canny(blurred_frame, 50, 150)

    # Get predictions from the model
    # resized_frame = cv2.resize(blurred_frame, (640, 480))
    # frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    dimmed_frame = cv2.convertScaleAbs(frame, alpha=0.8, beta=0)
    results = model(dimmed_frame, conf=IDENTIFICATION_CONFIDENCE)

    detections = results[0].boxes if results[0].boxes is not None else []

    frame = results[0].plot()

    # Detect and update stumps bounding box
    CURRENT_STUMPS_BBOX = detect_and_update_stumps(detections, CURRENT_STUMPS_BBOX, frame.shape)
    # if CURRENT_STUMPS_BBOX:
    #     CURRENT_STUMPS_BBOX = refine_stumps_with_edges(frame, CURRENT_STUMPS_BBOX)
    # print(f"Detected Stumps: {CURRENT_STUMPS_BBOX}")

    bat_box = None
    pad_box = None
    ball_detected = False

    # Process detections made in the frame
    for det in detections:
        class_id = int(det.cls[0])
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if class_id == 0:  # Ball detected
            ball_tracker.track_ball((cx, cy))
            ball_detected = True
            # print(f"Ball Box: {cx}, {cy}")  # Debug print
        elif class_id == 1:  # Bat
            bat_box = (x1, y1, x2, y2)
        elif class_id == 3:  # Pads
            pad_box = (x1, y1, x2, y2)
        
    # if not ball_detected:
    #     ball_tracker.predict_without_detection()
    
    # results = model(frame)[0]

    # Draw ball path and mark impact point
    ball_tracker.draw_path(frame)
    ball_tracker.mark_impact_point(frame)

    # LBW Logic
    if ball_tracker.impact_point != None:
        lbw = check_lbw(ball_tracker.get_impact_point(), pad_box=pad_box, bat_box=bat_box, stumps_box=CURRENT_STUMPS_BBOX)
        if lbw:
            cv2.putText(frame, "LBW!", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "NOT OUT!", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # Draw stumps bounding box
    if CURRENT_STUMPS_BBOX:
        cv2.rectangle(frame, (CURRENT_STUMPS_BBOX[0], CURRENT_STUMPS_BBOX[1]),
                      (CURRENT_STUMPS_BBOX[2], CURRENT_STUMPS_BBOX[3]), (0, 255, 0), 2)

    # Write processed frame to output
    out.write(frame)
    cv2.imshow("LBW Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
out.release()
cv2.destroyAllWindows()

