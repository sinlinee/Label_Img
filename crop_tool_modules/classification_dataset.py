import json
import os
import shutil
import uuid

from .constants import CLASS_MANIFEST_NAME
from .common_utils import safe_path_part
from .image_io import read_image
from .image_io import save_image


def flat_class_name(class_path):
    return "__".join(safe_path_part(part) for part in class_path.split("/") if part)


def save_crop(output_dir, input_dir, image_rel, image, bounds, class_path):
    saved, result = create_crop_record(output_dir, input_dir, image_rel, image, bounds, class_path)
    if not saved:
        return False, result

    update_manifest(output_dir, result)
    refresh_classification_exports(output_dir, input_dir)

    return True, f"状态：已保存分类模型样本 {os.path.join(output_dir, result['imagefolder_crop'])}，并更新分类导出数据。"


def create_crop_record(output_dir, input_dir, image_rel, image, bounds, class_path):
    class_parts = [part for part in class_path.split("/") if part]
    if not class_parts:
        return False, "状态：请先在分类树中选择一个分类，再保存。"

    x1, y1, x2, y2 = bounds
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return False, "状态：裁剪区域无效，保存失败。"

    tree_dir = os.path.join(output_dir, "classification", "tree", *(safe_path_part(part) for part in class_parts))
    flat_name = flat_class_name(class_path)
    imagefolder_dir = os.path.join(output_dir, "classification", "imagefolder", flat_name)
    os.makedirs(tree_dir, exist_ok=True)
    os.makedirs(imagefolder_dir, exist_ok=True)

    safe_rel = image_rel.replace(os.sep, "_") if image_rel else "0"
    file_name = f"{os.path.splitext(safe_rel)[0]}_{x1}_{y1}.jpg"
    tree_path = os.path.join(tree_dir, file_name)
    imagefolder_path = os.path.join(imagefolder_dir, file_name)
    if not save_image(tree_path, crop):
        return False, f"状态：保存失败 {tree_path}"
    if not save_image(imagefolder_path, crop):
        return False, f"状态：保存失败 {imagefolder_path}"

    source_image = os.path.join(input_dir, image_rel) if image_rel else ""
    record = {
        "id": uuid.uuid4().hex,
        "image_rel": image_rel,
        "source_image": source_image,
        "bbox_xyxy": [x1, y1, x2, y2],
        "class_path": class_path,
        "flat_class": flat_name,
        "tree_crop": os.path.relpath(tree_path, output_dir),
        "imagefolder_crop": os.path.relpath(imagefolder_path, output_dir),
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
    }
    return True, record


def manifest_path(output_dir):
    return os.path.join(output_dir, CLASS_MANIFEST_NAME)


def load_manifest(output_dir):
    path = manifest_path(output_dir)
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


def write_manifest(output_dir, records):
    os.makedirs(output_dir, exist_ok=True)
    with open(manifest_path(output_dir), "w", encoding="utf-8", newline="\n") as file_obj:
        for item in records:
            file_obj.write(json.dumps(item, ensure_ascii=False) + "\n")


def record_key(record):
    if record.get("id"):
        return ("id", record.get("id"))
    return (
        "legacy",
        record.get("image_rel"),
        tuple(record.get("bbox_xyxy", [])),
        record.get("class_path"),
    )


def update_manifest(output_dir, record):
    records = load_manifest(output_dir)
    new_key = record_key(record)
    records = [item for item in records if record_key(item) != new_key]
    records.append(record)
    write_manifest(output_dir, records)

def safe_remove_output_file(output_dir, rel_path):
    if not rel_path:
        return
    output_abs = os.path.abspath(output_dir)
    target = os.path.abspath(os.path.join(output_dir, rel_path))
    if not (target == output_abs or target.startswith(output_abs + os.sep)):
        return
    if os.path.isfile(target):
        os.remove(target)


def remove_record_outputs(output_dir, record):
    safe_remove_output_file(output_dir, record.get("tree_crop"))
    safe_remove_output_file(output_dir, record.get("imagefolder_crop"))


def delete_manifest_record(output_dir, input_dir, record):
    target_key = record_key(record)
    records = []
    removed = False
    for item in load_manifest(output_dir):
        if not removed and record_key(item) == target_key:
            remove_record_outputs(output_dir, item)
            removed = True
            continue
        records.append(item)
    if not removed:
        return False
    write_manifest(output_dir, records)
    refresh_classification_exports(output_dir, input_dir)
    return True


def update_manifest_record(output_dir, input_dir, record, image, bounds, class_path):
    if len(bounds) != 4:
        return False, "状态：历史标注坐标无效，无法修改。"

    saved, result = create_crop_record(
        output_dir,
        input_dir,
        record.get("image_rel"),
        image,
        tuple(int(value) for value in bounds),
        class_path,
    )
    if not saved:
        return False, result

    target_key = record_key(record)
    records = []
    replaced = False
    for item in load_manifest(output_dir):
        if not replaced and record_key(item) == target_key:
            remove_record_outputs(output_dir, item)
            records.append(result)
            replaced = True
        else:
            records.append(item)
    if not replaced:
        records.append(result)

    write_manifest(output_dir, records)
    refresh_classification_exports(output_dir, input_dir)
    return True, result


def update_manifest_record_class(output_dir, input_dir, record, image, class_path):
    bounds = record.get("bbox_xyxy") or []
    return update_manifest_record(output_dir, input_dir, record, image, bounds, class_path)


def refresh_classification_exports(output_dir, input_dir):
    save_class_index(output_dir)
    yolo_root = os.path.join(output_dir, "classification_yolo")
    output_abs = os.path.abspath(output_dir)
    yolo_abs = os.path.abspath(yolo_root)
    if os.path.isdir(yolo_root) and yolo_abs.startswith(output_abs + os.sep):
        shutil.rmtree(yolo_root)
    export_yolo_dataset(output_dir, input_dir)


def save_class_index(output_dir):
    records = load_manifest(output_dir)
    class_map = {}
    for record in records:
        flat_class = record.get("flat_class")
        class_path = record.get("class_path")
        if flat_class and class_path:
            class_map[flat_class] = class_path

    class_root = os.path.join(output_dir, "classification")
    os.makedirs(class_root, exist_ok=True)
    with open(os.path.join(class_root, "classes.txt"), "w", encoding="utf-8", newline="\n") as file_obj:
        for flat_class in sorted(class_map):
            file_obj.write(f"{flat_class}\t{class_map[flat_class]}\n")
    with open(os.path.join(class_root, "class_map.json"), "w", encoding="utf-8") as file_obj:
        json.dump(class_map, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")


def export_yolo_dataset(output_dir, input_dir):
    records = [
        record for record in load_manifest(output_dir)
        if record.get("image_rel") and record.get("bbox_xyxy") and record.get("class_path")
    ]
    if not records:
        return

    class_names = []
    for record in records:
        class_name = record["class_path"]
        if class_name not in class_names:
            class_names.append(class_name)

    yolo_root = os.path.join(output_dir, "classification_yolo")
    image_root = os.path.join(yolo_root, "images", "train")
    label_root = os.path.join(yolo_root, "labels", "train")
    os.makedirs(image_root, exist_ok=True)
    os.makedirs(label_root, exist_ok=True)

    labels_by_image = {}
    for record in records:
        image_rel = record["image_rel"]
        source_image = record.get("source_image") or os.path.join(input_dir, image_rel)
        target_image = os.path.join(image_root, image_rel)
        os.makedirs(os.path.dirname(target_image), exist_ok=True)
        if os.path.exists(source_image):
            source_abs = os.path.normcase(os.path.abspath(source_image))
            target_abs = os.path.normcase(os.path.abspath(target_image))
            if source_abs != target_abs:
                shutil.copy2(source_image, target_image)

        width = float(record.get("image_width") or 0)
        height = float(record.get("image_height") or 0)
        if width <= 0 or height <= 0:
            source = read_image(source_image) if os.path.exists(source_image) else None
            if source is None:
                continue
            height, width = source.shape[:2]

        x1, y1, x2, y2 = [float(value) for value in record["bbox_xyxy"]]
        class_id = class_names.index(record["class_path"])
        xc = ((x1 + x2) / 2) / width
        yc = ((y1 + y2) / 2) / height
        box_w = (x2 - x1) / width
        box_h = (y2 - y1) / height
        labels_by_image.setdefault(image_rel, []).append(
            f"{class_id} {xc:.6f} {yc:.6f} {box_w:.6f} {box_h:.6f}"
        )

    for image_rel, lines in labels_by_image.items():
        rel_stem = os.path.splitext(image_rel)[0] + ".txt"
        label_path = os.path.join(label_root, rel_stem)
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
        for idx, class_name in enumerate(class_names):
            safe_name = class_name.replace("'", "''")
            file_obj.write(f"  {idx}: '{safe_name}'\n")
