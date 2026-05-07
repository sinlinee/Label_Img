import json
import os
import shutil
import uuid

from .constants import BUSINESS_MANIFEST_NAME
from .common_utils import safe_path_part
from .yolo_utils import box_to_yolo_line


def business_manifest_path(output_dir):
    return os.path.join(output_dir, BUSINESS_MANIFEST_NAME)


def load_business_manifest(output_dir):
    path = business_manifest_path(output_dir)
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


def write_business_manifest(output_dir, records):
    os.makedirs(output_dir, exist_ok=True)
    with open(business_manifest_path(output_dir), "w", encoding="utf-8", newline="\n") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")


def business_record_key(record):
    if record.get("id"):
        return ("id", record["id"])
    return (
        "legacy",
        record.get("image_rel"),
        tuple(record.get("bbox_xyxy", [])),
        record.get("component_type"),
        record.get("biz_name"),
    )


def add_business_record(output_dir, input_dir, image_rel, image, bounds, component_type, biz_name, note=""):
    component_type = component_type.strip()
    biz_name = biz_name.strip()
    if not component_type:
        return False, "状态：请先填写 component_type。"
    if not biz_name:
        return False, "状态：请先填写 biz_name。"

    x1, y1, x2, y2 = [int(value) for value in bounds]
    if x2 <= x1 or y2 <= y1:
        return False, "状态：业务标注框无效。"
    record = {
        "id": uuid.uuid4().hex,
        "image_rel": image_rel,
        "source_image": os.path.join(input_dir, image_rel) if image_rel else "",
        "component_type": component_type,
        "biz_name": biz_name,
        "note": str(note).strip(),
        "bbox_xyxy": [x1, y1, x2, y2],
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
    }
    records = load_business_manifest(output_dir)
    records.append(record)
    write_business_manifest(output_dir, records)
    refresh_business_exports(output_dir, input_dir)
    return True, record


def delete_business_record(output_dir, input_dir, record):
    target_key = business_record_key(record)
    records = []
    removed = False
    for item in load_business_manifest(output_dir):
        if not removed and business_record_key(item) == target_key:
            removed = True
            continue
        records.append(item)
    if not removed:
        return False
    write_business_manifest(output_dir, records)
    refresh_business_exports(output_dir, input_dir)
    return True


def update_business_record(output_dir, input_dir, record, component_type, biz_name, note=""):
    component_type = component_type.strip()
    biz_name = biz_name.strip()
    if not component_type:
        return False, "状态：请先填写 component_type。"
    if not biz_name:
        return False, "状态：请先填写 biz_name。"
    target_key = business_record_key(record)
    records = []
    updated = None
    for item in load_business_manifest(output_dir):
        if updated is None and business_record_key(item) == target_key:
            item = dict(item)
            item["component_type"] = component_type
            item["biz_name"] = biz_name
            item["note"] = str(note).strip()
            updated = item
        records.append(item)
    if updated is None:
        return False, "状态：未找到要更新的业务标注。"
    write_business_manifest(output_dir, records)
    refresh_business_exports(output_dir, input_dir)
    return True, updated


def refresh_business_exports(output_dir, input_dir):
    records = [
        record for record in load_business_manifest(output_dir)
        if record.get("image_rel") and record.get("bbox_xyxy") and record.get("component_type")
    ]
    business_root = os.path.join(output_dir, "business")
    yolo_root = os.path.join(output_dir, "business_yolo")
    by_component_root = os.path.join(output_dir, "business_by_component")
    output_abs = os.path.abspath(output_dir)
    yolo_abs = os.path.abspath(yolo_root)
    if os.path.isdir(yolo_root) and yolo_abs.startswith(output_abs + os.sep):
        shutil.rmtree(yolo_root)
    by_component_abs = os.path.abspath(by_component_root)
    if os.path.isdir(by_component_root) and by_component_abs.startswith(output_abs + os.sep):
        shutil.rmtree(by_component_root)
    os.makedirs(business_root, exist_ok=True)

    class_names = []
    biz_rows = []
    for record in records:
        component_type = record["component_type"]
        if component_type not in class_names:
            class_names.append(component_type)
        biz_rows.append(
            {
                "image_rel": record.get("image_rel", ""),
                "component_type": component_type,
                "biz_name": record.get("biz_name", ""),
                "note": record.get("note", ""),
                "bbox_xyxy": record.get("bbox_xyxy", []),
            }
        )

    with open(os.path.join(business_root, "business_components.json"), "w", encoding="utf-8") as file_obj:
        json.dump(biz_rows, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
    with open(os.path.join(business_root, "component_types.txt"), "w", encoding="utf-8", newline="\n") as file_obj:
        for name in class_names:
            file_obj.write(f"{name}\n")

    if not records:
        return

    export_combined_yolo(records, class_names, yolo_root, input_dir)
    export_component_yolo_datasets(records, class_names, by_component_root, input_dir)


def copy_record_image(record, input_dir, image_root):
    image_rel = record["image_rel"]
    source_image = record.get("source_image") or os.path.join(input_dir, image_rel)
    target_image = os.path.join(image_root, image_rel)
    os.makedirs(os.path.dirname(target_image), exist_ok=True)
    if os.path.exists(source_image):
        source_abs = os.path.normcase(os.path.abspath(source_image))
        target_abs = os.path.normcase(os.path.abspath(target_image))
        if source_abs != target_abs:
            shutil.copy2(source_image, target_image)


def record_box_to_line(record, class_id):
    width = float(record.get("image_width") or 0)
    height = float(record.get("image_height") or 0)
    if width <= 0 or height <= 0:
        return ""
    box = {
        "class_id": class_id,
        "x1": record["bbox_xyxy"][0],
        "y1": record["bbox_xyxy"][1],
        "x2": record["bbox_xyxy"][2],
        "y2": record["bbox_xyxy"][3],
    }
    return box_to_yolo_line(box, width, height)


def write_yolo_dataset_files(yolo_root, class_names, labels_by_image):
    label_root = os.path.join(yolo_root, "labels", "train")
    os.makedirs(label_root, exist_ok=True)
    for image_rel, lines in labels_by_image.items():
        label_path = os.path.join(label_root, os.path.splitext(image_rel)[0] + ".txt")
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        with open(label_path, "w", encoding="utf-8", newline="\n") as file_obj:
            file_obj.write("\n".join(lines))
            file_obj.write("\n")

    with open(os.path.join(yolo_root, "classes.txt"), "w", encoding="utf-8", newline="\n") as file_obj:
        for class_name in class_names:
            file_obj.write(f"{class_name}\n")
    with open(os.path.join(yolo_root, "dataset.yaml"), "w", encoding="utf-8", newline="\n") as file_obj:
        file_obj.write("path: .\n")
        file_obj.write("train: images/train\n")
        file_obj.write("val: images/train\n")
        file_obj.write("names:\n")
        for index, class_name in enumerate(class_names):
            safe_name = class_name.replace("'", "''")
            file_obj.write(f"  {index}: '{safe_name}'\n")


def export_combined_yolo(records, class_names, yolo_root, input_dir):
    image_root = os.path.join(yolo_root, "images", "train")
    os.makedirs(image_root, exist_ok=True)

    labels_by_image = {}
    for record in records:
        image_rel = record["image_rel"]
        copy_record_image(record, input_dir, image_root)
        line = record_box_to_line(record, class_names.index(record["component_type"]))
        if not line:
            continue
        labels_by_image.setdefault(image_rel, []).append(line)

    write_yolo_dataset_files(yolo_root, class_names, labels_by_image)


def export_component_yolo_datasets(records, class_names, by_component_root, input_dir):
    os.makedirs(by_component_root, exist_ok=True)
    for component_type in class_names:
        component_records = [record for record in records if record.get("component_type") == component_type]
        component_dir = os.path.join(by_component_root, safe_path_part(component_type))
        image_root = os.path.join(component_dir, "images", "train")
        os.makedirs(image_root, exist_ok=True)

        labels_by_image = {}
        biz_rows = []
        for record in component_records:
            image_rel = record["image_rel"]
            copy_record_image(record, input_dir, image_root)
            line = record_box_to_line(record, 0)
            if line:
                labels_by_image.setdefault(image_rel, []).append(line)
            biz_rows.append(
                {
                    "image_rel": image_rel,
                    "component_type": component_type,
                    "biz_name": record.get("biz_name", ""),
                    "note": record.get("note", ""),
                    "bbox_xyxy": record.get("bbox_xyxy", []),
                }
            )

        write_yolo_dataset_files(component_dir, [component_type], labels_by_image)
        with open(os.path.join(component_dir, "business_components.json"), "w", encoding="utf-8") as file_obj:
            json.dump(biz_rows, file_obj, ensure_ascii=False, indent=2)
            file_obj.write("\n")
