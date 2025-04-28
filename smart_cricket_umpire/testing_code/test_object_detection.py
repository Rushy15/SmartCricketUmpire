import cv2
from ultralytics import YOLO


MODEL_PATH = "models/model2_best.pt"

VIDEO_NUM = 11
VIDEO_PATH = f"outputs/testing_videos/{VIDEO_NUM}.mp4"

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

# # === Output video writer (optional) ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(f'output{VIDEO_NUM}.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # === Inference ===
    results = model(frame, conf=0.3)  # Adjust confidence threshold as needed

    # === Get YOLOv8 result (note: results is a list) ===
    boxes = results[0].boxes  # First (and only) result for one frame

    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            label = model.names[cls_id]  # Get class name

            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box

            # Draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # === Show / Save frame ===
    out.write(frame)
    cv2.imshow('Smart Umpire', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
 