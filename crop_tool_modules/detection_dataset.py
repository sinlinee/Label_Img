import os
import shutil


def label_path(output_dir, image_rel):
    if not image_rel:
        return os.path.join(output_dir, "labels", "train", "unknown.txt")
    rel_stem = os.path.splitext(image_rel)[0] + ".txt"
    return os.path.join(output_dir, "labels", "train", rel_stem)


def legacy_label_path(output_dir, image_rel):
    if not image_rel:
        return os.path.join(output_dir, "labels", "unknown.txt")
    rel_stem = os.path.splitext(image_rel)[0] + ".txt"
    return os.path.join(output_dir, "labels", rel_stem)


def dataset_image_path(output_dir, image_rel):
    if not image_rel:
        return os.path.join(output_dir, "images", "train", "unknown.jpg")
    return os.path.join(output_dir, "images", "train", image_rel)


def copy_image_to_dataset(output_dir, image_rel, source_path):
    if not source_path or not os.path.exists(source_path):
        return False, "", f"Status: source image not found: {source_path}"

    image_path = dataset_image_path(output_dir, image_rel)
    os.makedirs(os.path.dirname(image_path), exist_ok=True)

    source_abs = os.path.normcase(os.path.abspath(source_path))
    target_abs = os.path.normcase(os.path.abspath(image_path))
    if source_abs != target_abs:
        try:
            shutil.copy2(source_path, image_path)
        except OSError as exc:
            return False, image_path, f"Status: failed to copy image to YOLO dataset: {exc}"

    return True, image_path, ""


def load_labels_index(output_dir):
    labels_index = {}
    index_path = os.path.join(output_dir, "labels.txt")
    if not os.path.exists(index_path):
        return labels_index

    with open(index_path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            labels_index.setdefault(parts[0], []).append(parts[1])
    return labels_index


def save_labels_index(output_dir, labels_index):
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, "labels.txt")
    with open(index_path, "w", encoding="utf-8", newline="\n") as file_obj:
        for image_rel in sorted(labels_index):
            for yolo_line in labels_index[image_rel]:
                file_obj.write(f"{image_rel}\t{yolo_line}\n")


def save_detection_classes(output_dir, class_names):
    os.makedirs(output_dir, exist_ok=True)
    classes_path = os.path.join(output_dir, "classes.txt")
    with open(classes_path, "w", encoding="utf-8", newline="\n") as file_obj:
        for name in class_names:
            file_obj.write(f"{name}\n")

    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8", newline="\n") as file_obj:
        file_obj.write("path: .\n")
        file_obj.write("train: images/train\n")
        file_obj.write("val: images/train\n")
        file_obj.write("names:\n")
        for idx, name in enumerate(class_names):
            safe_name = name.replace("'", "''")
            file_obj.write(f"  {idx}: '{safe_name}'\n")
