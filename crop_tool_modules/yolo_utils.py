from .common_utils import clamp


def normalized_rect_to_xyxy(class_id, values, img_width, img_height, class_names):
    xc, yc, box_w, box_h = values
    x1 = int(round((xc - box_w / 2) * img_width))
    y1 = int(round((yc - box_h / 2) * img_height))
    x2 = int(round((xc + box_w / 2) * img_width))
    y2 = int(round((yc + box_h / 2) * img_height))
    x1 = clamp(x1, 0, img_width - 1)
    y1 = clamp(y1, 0, img_height - 1)
    x2 = clamp(x2, 1, img_width)
    y2 = clamp(y2, 1, img_height)
    if x2 <= x1 or y2 <= y1:
        return None

    class_name = class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"
    return {
        "class_id": class_id,
        "class_name": class_name,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "score": None,
    }


def box_to_yolo_line(box, img_width, img_height):
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    xc = ((x1 + x2) / 2) / img_width
    yc = ((y1 + y2) / 2) / img_height
    box_w = (x2 - x1) / img_width
    box_h = (y2 - y1) / img_height
    return f"{box['class_id']} {xc:.6f} {yc:.6f} {box_w:.6f} {box_h:.6f}"
