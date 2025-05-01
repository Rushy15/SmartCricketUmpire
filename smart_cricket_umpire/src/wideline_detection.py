# def box_area(self, box):
#     x1, y1, x2, y2 = box
#     return max(0, x2 - x1) * max(0, y2 - y1)

# def find_wide_box(detections, wide_class_id):
#     for det in detections:
#         class_id, conf, bbox = det
#         if class_id == wide_class_id:
#             return bbox
#     return None

# def decide_expand_direction(wide_box):
#     if not wide_box:
#         return True
    
#     x1, y1, x2, y2 = wide_box
#     width = x2 - x1
#     centre_x = (x1 + x2) / 2

#     wide_area = box_area(wide_box)

#     if centre_x > 0.5:
#         return True