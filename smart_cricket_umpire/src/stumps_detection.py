# stumps_detection.py

def expand_stump_box(box, scale_x=3, scale_y=1.0):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1

    new_width = int(width * scale_x)
    new_height = int(height * scale_y)

    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    x1_new = max(0, center_x - new_width // 2)
    y1_new = max(0, center_y - new_height // 2)
    x2_new = center_x + new_width // 2
    y2_new = center_y + new_height // 2

    return (x1_new, y1_new, x2_new, y2_new)


def detect_and_update_stumps(detections, current_stumps_bbox):
    for det in detections:
        class_id = int(det.cls[0])
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        
        if class_id == 4:  # stumps
            new_box = expand_stump_box((x1, y1, x2, y2))
            new_area = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1])
            
            if current_stumps_bbox:
                current_area = (current_stumps_bbox[2] - current_stumps_bbox[0]) * (current_stumps_bbox[3] - current_stumps_bbox[1])
                if new_area > current_area:
                    current_stumps_bbox = new_box
            else:
                current_stumps_bbox = new_box

    return current_stumps_bbox