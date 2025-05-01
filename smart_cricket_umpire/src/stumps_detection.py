# stumps_detection.py
import cv2

TARGET_RATIO = 3.0
LEFT_SCALE = 1.0
RIGHT_SCALE = 3.0
SCALE_Y = 1.0

def expand_stump_box(box, img_width, img_height, expand_left=True):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1

    center_y = (y1 + y2) // 2

    if expand_left:
        x1_new = max(0, int(x1 - width * LEFT_SCALE))
        x2_new = int(x2 + width * RIGHT_SCALE)
    else:
        x1_new = max(0, int(x1 - width * RIGHT_SCALE))
        x2_new = int(x2 + width * LEFT_SCALE)

    new_height = int(height * SCALE_Y)
    y1_new = max(0, center_y - new_height // 2)
    y2_new = center_y + new_height // 2

    return (x1_new, y1_new, x2_new, y2_new)


def detect_and_update_stumps(detections, current_stumps_bbox, img_shape):
    img_height, img_width = img_shape[:2]
    best_box = None
    best_conf = 0
    best_area = 0

    for det in detections:
        class_id = int(det.cls[0])
        conf = float(det.conf[0])
        
        if class_id == 4: #and conf > 0.4:  # stumps
            best_conf = conf
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            new_box_area = (x2 - x1) * (y2 - y1)
            
            if new_box_area > best_area:
                best_box = (x1, y1, x2, y2)
                best_area = new_box_area
            
    if best_box:
        expanded_box = expand_stump_box(best_box, img_width, img_height)
        new_expanded_area = (expanded_box[2] - expanded_box[0]) * (expanded_box[3] - expanded_box[1])
        
        if current_stumps_bbox:
            current_area = (current_stumps_bbox[2] - current_stumps_bbox[0]) * (current_stumps_bbox[3] - current_stumps_bbox[1])
            
            if best_conf > 0.4 and new_expanded_area > current_area:
                current_stumps_bbox = expanded_box
            
            elif new_expanded_area > current_area:
                alpha = 0.7 # Smooth update using weighted average
                current_stumps_bbox = (
                    int(alpha * current_stumps_bbox[0] + (1 - alpha) * expanded_box[0]),
                    int(alpha * current_stumps_bbox[1] + (1 - alpha) * expanded_box[1]),
                    int(alpha * current_stumps_bbox[2] + (1 - alpha) * expanded_box[2]),
                    int(alpha * current_stumps_bbox[3] + (1 - alpha) * expanded_box[3]),
                )
        else:
            current_stumps_bbox = expanded_box

    return current_stumps_bbox


def refine_stumps_with_edges(frame, current_stumps_bbox):
    if not current_stumps_bbox:
        return current_stumps_bbox
    
    x1, y1, x2, y2 = current_stumps_bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

    stumps_crop = frame[y1:y2, x1:x2]
    if stumps_crop.size == 0:
        return current_stumps_bbox
    
    grey = cv2.cvtColor(stumps_crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        return (x1 + x, y1 + y, x1 + x + w, y1 + y + h)
    
    return current_stumps_bbox