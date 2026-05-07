import math
import os
import re

from PyQt5.QtWidgets import QFileDialog, QLineEdit, QPushButton

from .common_utils import clamp, copy_boxes
from .constants import BASE_DIR
from . import detection_dataset as _det_ds
from .detection_dataset import (
    copy_image_to_dataset,
    label_path,
    legacy_label_path,
    load_labels_index,
    save_labels_index,
)
from .yolo_utils import box_to_yolo_line, normalized_rect_to_xyxy


class DetectionMixin:

    def load_detection_classes_config(self):
        classes_path = os.path.join(self.output_dir, "classes.txt")
        if not os.path.exists(classes_path):
            return

        with open(classes_path, "r", encoding="utf-8") as file_obj:
            names = [line.strip() for line in file_obj if line.strip()]
        if names:
            self.det_class_names = names
            if self.saved_det_class_name in self.det_class_names:
                self.active_det_class_id = self.det_class_names.index(self.saved_det_class_name)
            else:
                self.active_det_class_id = clamp(self.active_det_class_id, 0, len(names) - 1)
            self.saved_det_class_name = self.det_class_names[self.active_det_class_id]

    def rebuild_detection_class_buttons(self):
        while self.det_class_grid.count():
            item = self.det_class_grid.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        self.det_class_buttons = []
        self.det_class_name_edits = []
        for idx, name in enumerate(self.det_class_names):
            button = QPushButton(str(idx))
            button.setCheckable(True)
            button.setChecked(idx == self.active_det_class_id)
            button.clicked.connect(lambda _, value=idx: self.set_detection_class(value))
            edit = QLineEdit(name)
            edit.setMinimumWidth(160)
            edit.cursorPositionChanged.connect(
                lambda _old, _new, value=idx: self.set_detection_class(value)
            )
            edit.textEdited.connect(lambda _text, value=idx: self.set_detection_class(value))
            edit.editingFinished.connect(
                lambda value=idx, widget=edit: self.rename_detection_class_from_edit(value, widget)
            )
            self.det_class_buttons.append(button)
            self.det_class_name_edits.append(edit)
            self.det_class_grid.addWidget(button, idx, 0)
            self.det_class_grid.addWidget(edit, idx, 1)
        self.refresh_detection_class_selection()

    def refresh_detection_class_selection(self):
        for idx, button in enumerate(getattr(self, "det_class_buttons", [])):
            button.setChecked(idx == self.active_det_class_id)
        for idx, edit in enumerate(getattr(self, "det_class_name_edits", [])):
            if idx == self.active_det_class_id:
                edit.setStyleSheet("QLineEdit { border: 2px solid #2563eb; font-weight: 600; }")
            else:
                edit.setStyleSheet("")

    def sync_detection_classes_text(self):
        if hasattr(self, "det_classes_edit"):
            self.det_classes_edit.setText(",".join(self.det_class_names))

    def apply_detection_classes(self):
        if hasattr(self, "det_classes_edit"):
            names = [name.strip() for name in self.det_classes_edit.text().split(",") if name.strip()]
        else:
            names = [edit.text().strip() for edit in self.det_class_name_edits if edit.text().strip()]
        if not names:
            self.set_status_message("状态：目标类别不能为空。")
            return

        old_names = list(self.det_class_names)
        self.det_class_names = names
        self.active_det_class_id = clamp(self.active_det_class_id, 0, len(names) - 1)
        for box in self.det_boxes:
            if box["class_name"] in self.det_class_names:
                box["class_id"] = self.det_class_names.index(box["class_name"])
            elif 0 <= box["class_id"] < len(self.det_class_names):
                box["class_name"] = self.det_class_names[box["class_id"]]
            elif box["class_name"] in old_names:
                box["class_id"] = self.det_class_names.index(names[0])
                box["class_name"] = names[0]
        self.rebuild_detection_class_buttons()
        self.sync_detection_classes_text()
        self.mark_detection_dirty()
        self.save_detection_classes()
        self.update_choice_text()
        self.update_detection_info()
        self.saved_det_class_name = self.det_class_names[self.active_det_class_id]
        self.save_app_settings()
        self.set_status_message(f"状态：已应用目标识别类别：{', '.join(self.det_class_names)}")

    def unique_detection_class_name(self, base_name="新类别"):
        if base_name not in self.det_class_names:
            return base_name
        index = 2
        while f"{base_name}{index}" in self.det_class_names:
            index += 1
        return f"{base_name}{index}"

    def add_detection_class_inline(self):
        class_name = self.unique_detection_class_name()
        self.det_class_names.append(class_name)
        self.active_det_class_id = len(self.det_class_names) - 1
        self.sync_detection_classes_text()
        self.rebuild_detection_class_buttons()
        self.save_detection_classes()
        self.update_choice_text()
        self.saved_det_class_name = class_name
        self.save_app_settings()
        if self.det_class_name_edits:
            edit = self.det_class_name_edits[-1]
            edit.setFocus()
            edit.selectAll()
        self.set_status_message("状态：已新增目标类别，可直接修改名称。")

    def rename_detection_class_from_edit(self, class_id, edit):
        if not (0 <= class_id < len(self.det_class_names)):
            return

        old_name = self.det_class_names[class_id]
        new_name = edit.text().strip()
        if not new_name:
            edit.setText(old_name)
            self.set_status_message("状态：类别名称不能为空。")
            return

        if new_name in self.det_class_names and self.det_class_names.index(new_name) != class_id:
            edit.setText(old_name)
            self.set_status_message(f"状态：类别 {new_name} 已存在。")
            return

        if new_name == old_name:
            edit.setText(old_name)
            return

        self.det_class_names[class_id] = new_name
        self.rename_detection_boxes(class_id, old_name, new_name, self.det_boxes)
        for boxes in self.det_cache.values():
            self.rename_detection_boxes(class_id, old_name, new_name, boxes)
        self.sync_detection_classes_text()
        self.save_detection_classes()
        self.active_det_class_id = class_id
        self.saved_det_class_name = new_name
        self.refresh_detection_class_selection()
        self.update_choice_text()
        self.update_detection_overlay()
        self.save_app_settings()
        self.set_status_message(f"状态：已重命名目标类别 {old_name} -> {new_name}。")

    def rename_detection_boxes(self, class_id, old_name, new_name, boxes):
        for box in boxes:
            if box.get("class_id") == class_id or box.get("class_name") == old_name:
                box["class_id"] = class_id
                box["class_name"] = new_name

    def delete_detection_class(self):
        if not self.det_class_names:
            return
        if len(self.det_class_names) <= 1:
            self.set_status_message("状态：至少保留一个目标类别。")
            return

        deleted_id = self.active_det_class_id
        deleted_name = self.det_class_names.pop(deleted_id)
        self.det_boxes, removed_current = self.remap_boxes_after_class_delete(
            self.det_boxes, deleted_id, deleted_name
        )
        removed_cached = 0
        for image_rel, boxes in list(self.det_cache.items()):
            remapped, removed = self.remap_boxes_after_class_delete(boxes, deleted_id, deleted_name)
            self.det_cache[image_rel] = remapped
            removed_cached += removed

        self.active_det_class_id = clamp(deleted_id, 0, len(self.det_class_names) - 1)
        self.saved_det_class_name = self.det_class_names[self.active_det_class_id]
        self.sync_detection_classes_text()
        self.rebuild_detection_class_buttons()
        self.save_detection_classes()
        if removed_current:
            self.mark_detection_dirty()
        self.update_choice_text()
        self.update_detection_overlay()
        self.save_app_settings()
        removed_total = removed_current + removed_cached
        suffix = f"，并移除内存中 {removed_total} 个该类别框" if removed_total else ""
        self.set_status_message(f"状态：已删除目标类别 {deleted_name}{suffix}。")

    def remap_boxes_after_class_delete(self, boxes, deleted_id, deleted_name):
        remapped = []
        removed = 0
        for box in boxes:
            class_id = int(box.get("class_id", -1))
            class_name = box.get("class_name", "")
            if class_id == deleted_id or class_name == deleted_name:
                removed += 1
                continue
            if class_id > deleted_id:
                box["class_id"] = class_id - 1
            if 0 <= box.get("class_id", -1) < len(self.det_class_names):
                box["class_name"] = self.det_class_names[box["class_id"]]
            remapped.append(box)
        return remapped, removed

    def set_detection_class(self, class_id):
        if not (0 <= class_id < len(self.det_class_names)):
            return
        self.active_det_class_id = class_id
        self.saved_det_class_name = self.det_class_names[class_id]
        self.refresh_detection_class_selection()
        self.update_choice_text()
        self.save_app_settings()

    def find_or_add_detection_class(self, class_name):
        class_name = str(class_name).strip() or "object"
        if class_name not in self.det_class_names:
            self.det_class_names.append(class_name)
            self.sync_detection_classes_text()
            self.rebuild_detection_class_buttons()
        return self.det_class_names.index(class_name)

    def add_detection_box(self, rect):
        if self.img is None or not self.det_class_names:
            return

        img_height, img_width = self.img.shape[:2]
        x1 = clamp(rect.left(), 0, img_width - 1)
        y1 = clamp(rect.top(), 0, img_height - 1)
        x2 = clamp(rect.left() + rect.width(), 1, img_width)
        y2 = clamp(rect.top() + rect.height(), 1, img_height)
        if x2 <= x1 or y2 <= y1:
            return

        class_name = self.det_class_names[self.active_det_class_id]
        self.det_boxes.append(
            {
                "class_id": self.active_det_class_id,
                "class_name": class_name,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "score": None,
            }
        )
        self.mark_detection_dirty()
        self.update_detection_overlay()
        self.set_status_message(f"状态：已新增 {class_name} 框。")

    def select_detection_box(self, index):
        self.image_label.selected_box_index = index
        if index >= 0 and index < len(self.det_boxes):
            box = self.det_boxes[index]
            self.selection_info.setText(
                f"检测框：{index + 1}/{len(self.det_boxes)} {box['class_name']} "
                f"({box['x1']}, {box['y1']})-({box['x2']}, {box['y2']})"
            )
        else:
            self.selection_info.setText("目标识别：未选中框")
        self.image_label.update()

    def delete_detection_box(self, index):
        if index < 0 or index >= len(self.det_boxes):
            return
        removed = self.det_boxes.pop(index)
        self.image_label.selected_box_index = -1
        self.mark_detection_dirty()
        self.update_detection_overlay()
        self.set_status_message(f"状态：已删除 {removed['class_name']} 框。")

    def delete_last_detection_box(self):
        if not self.det_boxes:
            self.set_status_message("状态：当前图片没有可删除的检测框。")
            return
        self.delete_detection_box(len(self.det_boxes) - 1)

    def clear_detection_boxes(self):
        self.det_boxes = []
        self.image_label.selected_box_index = -1
        self.mark_detection_dirty()
        self.update_detection_overlay()
        self.set_status_message("状态：已清空当前图片检测框。")

    def mark_detection_dirty(self):
        if self.current_image_rel:
            self.det_cache[self.current_image_rel] = copy_boxes(self.det_boxes)
        self.det_dirty = True

    def load_current_detection_labels(self):
        if self.img is None or not self.current_image_rel:
            return

        if self.current_image_rel in self.det_cache:
            self.det_boxes = copy_boxes(self.det_cache[self.current_image_rel])
            self.det_dirty = False
            self.update_detection_overlay()
            return

        lbl_path = self.current_label_path()
        lbl_legacy = self.current_legacy_label_path()
        if not os.path.exists(lbl_path) and os.path.exists(lbl_legacy):
            lbl_path = lbl_legacy
        boxes = []
        if os.path.exists(lbl_path):
            img_height, img_width = self.img.shape[:2]
            with open(lbl_path, "r", encoding="utf-8") as file_obj:
                for line in file_obj:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        class_id = int(float(parts[0]))
                        values = [float(value) for value in parts[1:5]]
                    except ValueError:
                        continue
                    box = normalized_rect_to_xyxy(
                        class_id, values, img_width, img_height, self.det_class_names
                    )
                    if box:
                        boxes.append(box)

        self.det_boxes = boxes
        self.det_cache[self.current_image_rel] = copy_boxes(self.det_boxes)
        self.det_dirty = False
        self.update_detection_overlay()

    def current_label_path(self):
        return label_path(self.output_dir, self.current_image_rel)

    def current_legacy_label_path(self):
        return legacy_label_path(self.output_dir, self.current_image_rel)

    def save_current_dataset_image(self):
        return copy_image_to_dataset(self.output_dir, self.current_image_rel, self.current_image_path())

    def save_current_labels(self):
        if self.img is None or not self.current_image_rel:
            return False, "状态：当前没有可保存的图片。"

        img_height, img_width = self.img.shape[:2]
        lbl_path = self.current_label_path()
        os.makedirs(os.path.dirname(lbl_path), exist_ok=True)
        lines = [box_to_yolo_line(box, img_width, img_height) for box in self.det_boxes]
        image_saved, image_path, image_message = self.save_current_dataset_image()
        if not image_saved:
            return False, image_message

        with open(lbl_path, "w", encoding="utf-8", newline="\n") as file_obj:
            file_obj.write("\n".join(lines))
            if lines:
                file_obj.write("\n")

        self.save_detection_classes()
        self.update_labels_txt_index(lines)
        self.det_cache[self.current_image_rel] = copy_boxes(self.det_boxes)
        self.det_dirty = False
        return True, f"状态：已保存目标识别图片 {image_path} 和检测框标注 {lbl_path}，并更新 labels.txt。"

    def save_dirty_detection_if_needed(self):
        if self.det_dirty:
            saved, message = self.save_current_labels()
            self.set_status_message(message)
            return saved
        return True

    def load_labels_txt_index(self):
        if self.labels_index_loaded:
            return
        self.labels_index = load_labels_index(self.output_dir)
        self.labels_index_loaded = True

    def update_labels_txt_index(self, yolo_lines):
        self.load_labels_txt_index()
        self.labels_index[self.current_image_rel] = yolo_lines
        save_labels_index(self.output_dir, self.labels_index)

    def save_detection_classes(self):
        _det_ds.save_detection_classes(self.output_dir, self.det_class_names)

    def choose_yolo_review_dir(self):
        selected_dir = QFileDialog.getExistingDirectory(
            self,
            "选择YOLO数据集根目录",
            self.yolo_review_dir or self.output_dir,
        )
        if not selected_dir:
            return
        self.yolo_review_dir = selected_dir
        self.yolo_review_dir_edit.setText(selected_dir)
        self.save_app_settings()

    def read_yolo_review_classes(self, dataset_dir):
        classes_path = os.path.join(dataset_dir, "classes.txt")
        if os.path.exists(classes_path):
            with open(classes_path, "r", encoding="utf-8") as file_obj:
                names = [line.strip() for line in file_obj if line.strip()]
            if names:
                return names

        yaml_path = os.path.join(dataset_dir, "dataset.yaml")
        if not os.path.exists(yaml_path):
            yaml_path = os.path.join(dataset_dir, "data.yaml")
        if not os.path.exists(yaml_path):
            return []

        names = []
        in_names = False
        with open(yaml_path, "r", encoding="utf-8") as file_obj:
            for raw_line in file_obj:
                line = raw_line.rstrip()
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped.startswith("names:"):
                    in_names = True
                    inline = stripped.split(":", 1)[1].strip()
                    if inline.startswith("[") and inline.endswith("]"):
                        return [
                            part.strip().strip("'\"")
                            for part in inline[1:-1].split(",")
                            if part.strip().strip("'\"")
                        ]
                    continue
                if in_names:
                    if not raw_line.startswith((" ", "\t")):
                        break
                    match = re.match(r"\s*(\d+)\s*:\s*['\"]?(.+?)['\"]?\s*$", line)
                    if match:
                        idx = int(match.group(1))
                        name = match.group(2).strip().strip("'\"")
                        while len(names) <= idx:
                            names.append("")
                        names[idx] = name
        return [name or f"class_{idx}" for idx, name in enumerate(names)]

    def open_yolo_review_dataset(self):
        dataset_dir = self.yolo_review_dir_edit.text().strip()
        if not dataset_dir:
            self.set_status_message("状态：请先选择YOLO数据集根目录。")
            return
        dataset_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(dataset_dir)))
        image_dir = os.path.join(dataset_dir, "images", "train")
        label_dir = os.path.join(dataset_dir, "labels", "train")
        if not os.path.isdir(image_dir) or not os.path.isdir(label_dir):
            self.set_status_message("状态：YOLO数据集需要包含 images/train 和 labels/train。")
            return

        if not self.save_dirty_detection_if_needed():
            return

        names = self.read_yolo_review_classes(dataset_dir)
        if names:
            self.det_class_names = names
            self.active_det_class_id = 0
            self.saved_det_class_name = names[0]

        self.yolo_review_dir = dataset_dir
        self.input_dir_value.setText(image_dir)
        self.output_dir_value.setText(dataset_dir)
        self.recursive_checkbox.setChecked(True)
        self.recursive_scan = True
        self.apply_directories_from_edits()
        self.set_mode("detect")
        self.set_status_message(f"状态：已打开YOLO标注数据集，可直接查看和修改：{dataset_dir}")

    def choose_model_path(self):
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "选择YOLO模型",
            self.model_path or BASE_DIR,
            "YOLO model (*.pt *.onnx *.engine);;All files (*.*)",
        )
        if selected:
            self.model_path = selected
            self.model_path_edit.setText(selected)

    def load_yolo_model(self):
        self.model_path = self.model_path_edit.text().strip()
        if not self.model_path or not os.path.exists(self.model_path):
            self.set_status_message("状态：YOLO模型路径不存在。")
            return False

        try:
            from ultralytics import YOLO
        except ImportError:
            self.set_status_message("状态：未安装 ultralytics，无法加载YOLO模型。请先安装 ultralytics。")
            return False

        try:
            self.yolo_model = YOLO(self.model_path)
            names = getattr(self.yolo_model, "names", {})
            self.yolo_model_names = names if isinstance(names, dict) else dict(enumerate(names))
        except Exception as exc:
            self.yolo_model = None
            self.set_status_message(f"状态：YOLO模型加载失败：{exc}")
            return False

        self.set_status_message(f"状态：已加载YOLO模型 {self.model_path}")
        return True

    def auto_detect_current_image(self):
        if self.img is None or not self.current_image_rel:
            self.set_status_message("状态：当前没有可检测的图片。")
            return

        if self.yolo_model is None and not self.load_yolo_model():
            return

        image_path = self.current_image_path()
        try:
            results = self.yolo_model(image_path, conf=float(self.conf_spin.value()), verbose=False)
        except Exception as exc:
            self.set_status_message(f"状态：自动检测失败：{exc}")
            return

        if self.auto_replace_checkbox.isChecked():
            self.det_boxes = []

        added = 0
        for result in results:
            if not hasattr(result, "boxes") or result.boxes is None:
                continue
            for det in result.boxes:
                xyxy = det.xyxy[0].detach().cpu().numpy().tolist()
                class_id_from_model = int(det.cls[0].detach().cpu().item())
                score = float(det.conf[0].detach().cpu().item()) if det.conf is not None else None
                class_name = str(self.yolo_model_names.get(class_id_from_model, f"class_{class_id_from_model}"))
                class_id = self.find_or_add_detection_class(class_name)
                self.det_boxes.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "x1": int(round(xyxy[0])),
                        "y1": int(round(xyxy[1])),
                        "x2": int(round(xyxy[2])),
                        "y2": int(round(xyxy[3])),
                        "score": score,
                    }
                )
                added += 1

        self.mark_detection_dirty()
        self.update_detection_overlay()
        self.set_status_message(f"状态：自动检测完成，新增 {added} 个框。")

    def toggle_connection_overlay(self):
        self.image_label.set_show_connections(self.show_links_checkbox.isChecked())
        self.update_detection_overlay()

    def class_id_by_name(self, name):
        name = name.strip()
        if not name:
            return None
        for idx, class_name in enumerate(self.det_class_names):
            if class_name == name:
                return idx
        return None

    def box_center(self, box):
        return (box["x1"] + box["x2"]) / 2, (box["y1"] + box["y2"]) / 2

    def calculate_connections(self):
        source_id = self.class_id_by_name(self.link_source_edit.text())
        target_id = self.class_id_by_name(self.link_target_edit.text())
        if source_id is None or target_id is None:
            return [], []

        target_indexes = [
            index for index, box in enumerate(self.det_boxes) if box["class_id"] == target_id
        ]
        source_indexes = [
            index for index, box in enumerate(self.det_boxes) if box["class_id"] == source_id
        ]
        max_distance = self.max_distance_spin.value()
        connections = []
        warnings = []

        for source_index in source_indexes:
            if not target_indexes:
                warnings.append(f"未匹配：{self.det_boxes[source_index]['class_name']} #{source_index + 1} 没有目标 {self.link_target_edit.text()}")
                continue

            sx, sy = self.box_center(self.det_boxes[source_index])
            nearest_index = None
            nearest_distance = None
            for target_index in target_indexes:
                tx, ty = self.box_center(self.det_boxes[target_index])
                distance = math.hypot(sx - tx, sy - ty)
                if nearest_distance is None or distance < nearest_distance:
                    nearest_distance = distance
                    nearest_index = target_index

            too_far = nearest_distance is not None and nearest_distance > max_distance
            connections.append((source_index, nearest_index, nearest_distance, too_far))
            if too_far:
                warnings.append(
                    f"距离过远：{self.det_boxes[source_index]['class_name']} #{source_index + 1} "
                    f"到 {self.det_boxes[nearest_index]['class_name']} #{nearest_index + 1} 为 {nearest_distance:.1f}px"
                )

        return connections, warnings

    def update_detection_overlay(self):
        self.image_label.set_detection_boxes(self.det_boxes)
        connections, _ = self.calculate_connections()
        self.image_label.set_connections(connections)
        self.update_detection_info()

    def update_detection_info(self):
        counts = {}
        for box in self.det_boxes:
            counts[box["class_name"]] = counts.get(box["class_name"], 0) + 1
        parts = [f"{name}:{count}" for name, count in counts.items()]
        suffix = "，".join(parts) if parts else "无"
        self.det_info.setText(f"检测框：{len(self.det_boxes)}（{suffix}）")

    def check_annotations(self):
        connections, warnings = self.calculate_connections()
        self.image_label.set_connections(connections)
        if warnings:
            preview = "；".join(warnings[:4])
            more = f"；另有 {len(warnings) - 4} 条" if len(warnings) > 4 else ""
            self.set_status_message(f"状态：标注检查发现问题：{preview}{more}")
        else:
            self.set_status_message("状态：标注检查通过。")
