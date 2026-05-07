import json
import math
import os
import shutil
import uuid

from .constants import POINTER_MANIFEST_NAME


def pointer_manifest_path(output_dir):
    return os.path.join(output_dir, POINTER_MANIFEST_NAME)


def load_pointer_manifest(output_dir):
    path = pointer_manifest_path(output_dir)
    if not os.path.exists(path):
        return []

    records = []
    with open(path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def write_pointer_manifest(output_dir, records):
    os.makedirs(output_dir, exist_ok=True)
    with open(pointer_manifest_path(output_dir), "w", encoding="utf-8", newline="\n") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")


def pointer_record_key(record):
    if record.get("id"):
        return ("id", record["id"])
    return (
        "legacy",
        record.get("image_rel"),
        tuple(record.get("center_xy", [])),
        tuple(record.get("tip_xy", [])),
    )


def angle_from_points(center, tip):
    dx = float(tip[0]) - float(center[0])
    dy = float(tip[1]) - float(center[1])
    return (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0


def angle_diff(a, b):
    return (float(a) - float(b) + 180.0) % 360.0 - 180.0


def circular_mean(angles):
    values = [float(angle) for angle in angles]
    if not values:
        return 0.0
    sin_sum = sum(math.sin(math.radians(angle)) for angle in values)
    cos_sum = sum(math.cos(math.radians(angle)) for angle in values)
    if abs(sin_sum) < 1e-9 and abs(cos_sum) < 1e-9:
        return sum(values) / len(values)
    return (math.degrees(math.atan2(sin_sum, cos_sum)) + 360.0) % 360.0


def optional_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def bbox_from_points(center, tip, image_width, image_height, padding):
    cx, cy = [int(value) for value in center]
    tx, ty = [int(value) for value in tip]
    x1 = max(0, min(cx, tx) - padding)
    y1 = max(0, min(cy, ty) - padding)
    x2 = min(image_width, max(cx, tx) + padding)
    y2 = min(image_height, max(cy, ty) + padding)
    if x2 <= x1:
        x2 = min(image_width, x1 + 1)
    if y2 <= y1:
        y2 = min(image_height, y1 + 1)
    return [int(x1), int(y1), int(x2), int(y2)]


def save_pointer_record(
    output_dir,
    input_dir,
    image_rel,
    image,
    center,
    tip,
    label="",
    text="",
    panel_id="",
    panel_type="",
    object_name="",
    object_type="rotary",
    gauge_value=None,
    gauge_min_value=None,
    gauge_max_value=None,
    bbox=None,
    bbox_padding=80,
):
    cx, cy = [int(value) for value in center]
    tx, ty = [int(value) for value in tip]
    if cx == tx and cy == ty:
        return False, "状态：指针中心和尖端不能是同一点。"
    normalized_panel_type = (panel_type or panel_id or "").strip()
    if not normalized_panel_type:
        return False, "状态：请先填写 panel_type，同一种面板的标注会合并成一套规则。"

    height, width = image.shape[:2]
    bbox_xyxy = bbox or bbox_from_points((cx, cy), (tx, ty), width, height, int(bbox_padding))
    angle_deg = angle_from_points((cx, cy), (tx, ty))
    normalized_type = str(object_type or "").strip()
    if not normalized_type:
        return False, "状态：请先填写角度对象种类。"
    record = {
        "id": uuid.uuid4().hex,
        "type": normalized_type,
        "object_type": normalized_type,
        "image_rel": image_rel,
        "source_image": os.path.join(input_dir, image_rel) if image_rel else "",
        "panel_type": normalized_panel_type,
        "panel_id": panel_id.strip(),
        "object_name": object_name.strip(),
        "label": label.strip(),
        "text": text.strip(),
        "gauge_value": optional_float(gauge_value),
        "gauge_min_value": optional_float(gauge_min_value),
        "gauge_max_value": optional_float(gauge_max_value),
        "bbox_xyxy": bbox_xyxy,
        "center_xy": [cx, cy],
        "tip_xy": [tx, ty],
        "keypoints": {
            "center": [cx, cy],
            "tip": [tx, ty],
        },
        "angle_deg": round(angle_deg, 3),
        "angle_rad": round(math.radians(angle_deg), 6),
        "image_width": int(width),
        "image_height": int(height),
    }
    records = load_pointer_manifest(output_dir)
    records.append(record)
    write_pointer_manifest(output_dir, records)
    refresh_pointer_exports(output_dir, input_dir)
    return True, record


def delete_pointer_record(output_dir, record, input_dir=""):
    target_key = pointer_record_key(record)
    records = []
    removed = False
    for item in load_pointer_manifest(output_dir):
        if not removed and pointer_record_key(item) == target_key:
            removed = True
            continue
        records.append(item)
    if not removed:
        return False
    write_pointer_manifest(output_dir, records)
    refresh_pointer_exports(output_dir, input_dir)
    return True


def update_pointer_record_metadata(
    output_dir,
    input_dir,
    record,
    label="",
    text="",
    panel_type="",
    object_name="",
    object_type="rotary",
    gauge_value=None,
    gauge_min_value=None,
    gauge_max_value=None,
):
    target_key = pointer_record_key(record)
    normalized_panel_type = (panel_type or "").strip()
    if not normalized_panel_type:
        return False, "状态：请先填写 panel_type。"

    records = []
    updated = None
    for item in load_pointer_manifest(output_dir):
        if updated is None and pointer_record_key(item) == target_key:
            item = dict(item)
            normalized_type = str(object_type or "").strip()
            if not normalized_type:
                return False, "状态：请先填写角度对象种类。"
            item["panel_type"] = normalized_panel_type
            item["panel_id"] = normalized_panel_type
            item["type"] = normalized_type
            item["object_type"] = normalized_type
            item["object_name"] = object_name.strip()
            item["label"] = label.strip()
            item["text"] = text.strip()
            item["gauge_value"] = optional_float(gauge_value)
            item["gauge_min_value"] = optional_float(gauge_min_value)
            item["gauge_max_value"] = optional_float(gauge_max_value)
            updated = item
        records.append(item)

    if updated is None:
        return False, "状态：未找到要更新的角度识别标注。"

    write_pointer_manifest(output_dir, records)
    refresh_pointer_exports(output_dir, input_dir)
    return True, updated


def record_semantic_name(record):
    return record.get("label") or record.get("text") or "未命名"


def record_object_type(record):
    object_type = record.get("object_type") or record.get("type") or "rotary"
    return object_type.strip() if isinstance(object_type, str) and object_type.strip() else "rotary"


def record_gauge_value(record):
    value = optional_float(record.get("gauge_value"))
    if value is not None:
        return value
    return optional_float(record_semantic_name(record))


def record_panel_id(record):
    return (record.get("panel_type") or record.get("panel_id") or "").strip()


def export_panel_rules(output_dir):
    grouped = {}
    for record in load_pointer_manifest(output_dir):
        panel_type = record_panel_id(record)
        if not panel_type:
            continue
        grouped.setdefault(panel_type, []).append(record)

    panel_types = {}
    for panel_type, records in sorted(grouped.items()):
        object_type = "gauge" if any(record_object_type(record) == "gauge" for record in records) else "rotary"
        panel = {"panel_type": panel_type, "type": object_type}

        if object_type == "gauge":
            calibration_by_value = {}
            min_values = []
            max_values = []
            for record in records:
                angle = float(record.get("angle_deg", 0))
                min_value = optional_float(record.get("gauge_min_value"))
                max_value = optional_float(record.get("gauge_max_value"))
                if min_value is not None:
                    min_values.append(min_value)
                if max_value is not None:
                    max_values.append(max_value)
                value = record_gauge_value(record)
                if value is not None:
                    calibration_by_value.setdefault(value, []).append(angle)

            calibration = [
                {"value": value, "angle": round(circular_mean(angles), 3)}
                for value, angles in sorted(calibration_by_value.items())
            ]
            calibration.sort(key=lambda item: item["value"])
            min_value = min(min_values) if min_values else None
            max_value = max(max_values) if max_values else None
            if min_value is None:
                min_value = calibration[0]["value"] if calibration else 0.0
            if max_value is None:
                max_value = calibration[-1]["value"] if calibration else 100.0

            min_angle = calibration[0]["angle"] if calibration else float(records[0].get("angle_deg", 0))
            max_angle = calibration[-1]["angle"] if calibration else float(records[-1].get("angle_deg", 0))
            for item in calibration:
                if abs(item["value"] - min_value) < 1e-9:
                    min_angle = item["angle"]
                if abs(item["value"] - max_value) < 1e-9:
                    max_angle = item["angle"]

            panel.update(
                {
                    "min_angle": round(float(min_angle), 3),
                    "max_angle": round(float(max_angle), 3),
                    "min_value": float(min_value),
                    "max_value": float(max_value),
                }
            )
            if calibration:
                panel["calibration"] = calibration
        else:
            position_angles = {}
            for record in records:
                name = record_semantic_name(record)
                if not name:
                    continue
                position_angles.setdefault(name, []).append(float(record.get("angle_deg", 0)))
            panel["positions"] = {
                name: round(circular_mean(angles), 3)
                for name, angles in sorted(position_angles.items())
            }

        panel_types[panel_type] = panel

    path = os.path.join(output_dir, "pointer_panel_rules.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "version": 1,
                "group_by": "panel_type",
                "panel_types": panel_types,
            },
            file_obj,
            ensure_ascii=False,
            indent=2,
        )
        file_obj.write("\n")


def export_pointer_yolo_pose(output_dir, input_dir):
    records = [
        record for record in load_pointer_manifest(output_dir)
        if record.get("image_rel") and record.get("bbox_xyxy") and record.get("center_xy") and record.get("tip_xy")
    ]
    yolo_root = os.path.join(output_dir, "pointer_yolo_pose")
    if os.path.isdir(yolo_root):
        shutil.rmtree(yolo_root)
    if not records:
        return

    image_root = os.path.join(yolo_root, "images", "train")
    label_root = os.path.join(yolo_root, "labels", "train")
    os.makedirs(image_root, exist_ok=True)
    os.makedirs(label_root, exist_ok=True)

    class_names = sorted({record_object_type(record) for record in records})
    class_id_by_name = {name: index for index, name in enumerate(class_names)}
    labels_by_image = {}
    for record in records:
        image_rel = record["image_rel"]
        source_image = record.get("source_image") or os.path.join(input_dir, image_rel)
        target_image = os.path.join(image_root, image_rel)
        os.makedirs(os.path.dirname(target_image), exist_ok=True)
        if source_image and os.path.exists(source_image):
            source_abs = os.path.normcase(os.path.abspath(source_image))
            target_abs = os.path.normcase(os.path.abspath(target_image))
            if source_abs != target_abs:
                shutil.copy2(source_image, target_image)

        width = float(record.get("image_width") or 0)
        height = float(record.get("image_height") or 0)
        if width <= 0 or height <= 0:
            continue
        x1, y1, x2, y2 = [float(value) for value in record["bbox_xyxy"]]
        cx, cy = [float(value) for value in record["center_xy"]]
        tx, ty = [float(value) for value in record["tip_xy"]]
        xc = ((x1 + x2) / 2) / width
        yc = ((y1 + y2) / 2) / height
        box_w = (x2 - x1) / width
        box_h = (y2 - y1) / height
        class_id = class_id_by_name.get(record_object_type(record), 0)
        line = (
            f"{class_id} {xc:.6f} {yc:.6f} {box_w:.6f} {box_h:.6f} "
            f"{cx / width:.6f} {cy / height:.6f} 2 "
            f"{tx / width:.6f} {ty / height:.6f} 2"
        )
        labels_by_image.setdefault(image_rel, []).append(line)

    for image_rel, lines in labels_by_image.items():
        label_path = os.path.join(label_root, os.path.splitext(image_rel)[0] + ".txt")
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        with open(label_path, "w", encoding="utf-8", newline="\n") as file_obj:
            file_obj.write("\n".join(lines))
            file_obj.write("\n")

    with open(os.path.join(yolo_root, "dataset.yaml"), "w", encoding="utf-8", newline="\n") as file_obj:
        file_obj.write("path: .\n")
        file_obj.write("train: images/train\n")
        file_obj.write("val: images/train\n")
        file_obj.write("kpt_shape: [2, 3]\n")
        file_obj.write("names:\n")
        for name, class_id in class_id_by_name.items():
            file_obj.write(f"  {class_id}: {name}\n")


def refresh_pointer_exports(output_dir, input_dir):
    export_panel_rules(output_dir)
    export_pointer_yolo_pose(output_dir, input_dir)
