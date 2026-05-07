import math

from PyQt5.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QLabel

from .common_utils import clamp
from .constants import MODE_BUSINESS, MODE_CROP, MODE_DETECT, MODE_POINTER


class ImageLabel(QLabel):
    selection_changed = pyqtSignal()
    crop_box_created = pyqtSignal(QRect)
    crop_box_deleted = pyqtSignal(int)
    detection_box_created = pyqtSignal(QRect)
    detection_box_selected = pyqtSignal(int)
    detection_box_deleted = pyqtSignal(int)
    business_box_created = pyqtSignal(QRect)
    business_box_deleted = pyqtSignal(int)
    pointer_created = pyqtSignal(QPoint, QPoint)
    pointer_point_clicked = pyqtSignal(QPoint)
    pointer_deleted = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.mode = MODE_CROP
        self.original_pixmap = None
        self.zoom_factor = 1.0
        self.start_point = None
        self.end_point = None
        self.selection_rect = QRect()
        self.crop_boxes = []
        self.crop_history_boxes = []
        self.det_boxes = []
        self.business_boxes = []
        self.connections = []
        self.pointer_records = []
        self.pointer_draft_center = None
        self.selected_box_index = -1
        self.show_connections = True
        self.crop_class_label = ""
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.setCursor(Qt.CrossCursor)
        self.setStyleSheet("background: #111827;")

    def set_mode(self, mode):
        self.mode = mode
        self.clear_selection()
        self.update()

    def set_image(self, pixmap):
        self.original_pixmap = pixmap
        self.update_scaled_pixmap()
        self.clear_selection()

    def set_zoom_factor(self, zoom_factor):
        self.zoom_factor = max(0.1, min(float(zoom_factor), 5.0))
        self.update_scaled_pixmap()
        self.update()

    def update_scaled_pixmap(self):
        if self.original_pixmap is None:
            self.clear()
            return

        if abs(self.zoom_factor - 1.0) < 0.001:
            pixmap = self.original_pixmap
        else:
            pixmap = self.original_pixmap.scaled(
                max(1, int(self.original_pixmap.width() * self.zoom_factor)),
                max(1, int(self.original_pixmap.height() * self.zoom_factor)),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        self.setPixmap(pixmap)
        self.resize(pixmap.size())

    def image_to_view_value(self, value):
        return int(round(value * self.zoom_factor))

    def image_to_view_rect(self, box):
        x1 = self.image_to_view_value(box["x1"])
        y1 = self.image_to_view_value(box["y1"])
        x2 = self.image_to_view_value(box["x2"])
        y2 = self.image_to_view_value(box["y2"])
        return QRect(x1, y1, x2 - x1, y2 - y1)

    def image_rect_to_view_rect(self, rect):
        rect = rect.normalized()
        x1 = self.image_to_view_value(rect.left())
        y1 = self.image_to_view_value(rect.top())
        x2 = self.image_to_view_value(rect.left() + rect.width())
        y2 = self.image_to_view_value(rect.top() + rect.height())
        return QRect(x1, y1, x2 - x1, y2 - y1)

    def view_to_image_point(self, point):
        if self.original_pixmap is None:
            return QPoint()
        x = int(round(point.x() / self.zoom_factor))
        y = int(round(point.y() / self.zoom_factor))
        x = clamp(x, 0, self.original_pixmap.width() - 1)
        y = clamp(y, 0, self.original_pixmap.height() - 1)
        return QPoint(x, y)

    def view_point_for_image_point(self, point):
        return QPoint(self.image_to_view_value(point.x()), self.image_to_view_value(point.y()))

    def set_detection_boxes(self, boxes):
        self.det_boxes = boxes
        if self.selected_box_index >= len(self.det_boxes):
            self.selected_box_index = -1
        self.update()

    def set_business_boxes(self, boxes):
        self.business_boxes = boxes
        self.update()

    def set_crop_boxes(self, boxes):
        self.crop_boxes = boxes
        self.update()

    def set_crop_history_boxes(self, boxes):
        self.crop_history_boxes = boxes
        self.update()

    def set_pointer_records(self, records):
        self.pointer_records = records
        self.update()

    def set_pointer_draft_center(self, point):
        self.pointer_draft_center = point
        self.update()

    def set_connections(self, connections):
        self.connections = connections
        self.update()

    def set_show_connections(self, checked):
        self.show_connections = checked
        self.update()

    def set_crop_class_label(self, label):
        self.crop_class_label = label or ""
        self.update()

    def clear_selection(self):
        self.start_point = None
        self.end_point = None
        self.selection_rect = QRect()
        self.selection_changed.emit()
        self.update()

    def has_selection(self):
        return (
            not self.selection_rect.isNull()
            and self.selection_rect.width() > 1
            and self.selection_rect.height() > 1
        )

    def get_selection_rect(self):
        return self.selection_rect.normalized()

    def clamp_point(self, point):
        if self.original_pixmap is None:
            return QPoint()
        return self.view_to_image_point(point)

    def box_at(self, point, boxes=None):
        boxes = self.det_boxes if boxes is None else boxes
        for index in range(len(boxes) - 1, -1, -1):
            box = boxes[index]
            if box["x1"] <= point.x() <= box["x2"] and box["y1"] <= point.y() <= box["y2"]:
                return index
        return -1

    def pointer_at(self, point):
        threshold = max(6.0, 8.0 / max(self.zoom_factor, 0.1))
        for index in range(len(self.pointer_records) - 1, -1, -1):
            record = self.pointer_records[index]
            center = record.get("center_xy") or []
            tip = record.get("tip_xy") or []
            if len(center) != 2 or len(tip) != 2:
                continue
            if self.point_segment_distance(point, QPoint(center[0], center[1]), QPoint(tip[0], tip[1])) <= threshold:
                return index
        return -1

    def point_segment_distance(self, point, start, end):
        px, py = point.x(), point.y()
        sx, sy = start.x(), start.y()
        ex, ey = end.x(), end.y()
        dx = ex - sx
        dy = ey - sy
        if dx == 0 and dy == 0:
            return math.hypot(px - sx, py - sy)
        t = ((px - sx) * dx + (py - sy) * dy) / float(dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        nearest_x = sx + t * dx
        nearest_y = sy + t * dy
        return math.hypot(px - nearest_x, py - nearest_y)

    def mousePressEvent(self, event):
        if not self.pixmap():
            return

        if self.mode == MODE_DETECT and event.button() == Qt.RightButton:
            index = self.box_at(self.clamp_point(event.pos()))
            if index >= 0:
                self.detection_box_deleted.emit(index)
            return

        if self.mode == MODE_BUSINESS and event.button() == Qt.RightButton:
            index = self.box_at(self.clamp_point(event.pos()), self.business_boxes)
            if index >= 0:
                self.business_box_deleted.emit(index)
            return

        if self.mode == MODE_CROP and event.button() == Qt.RightButton:
            index = self.box_at(self.clamp_point(event.pos()), self.crop_boxes)
            if index >= 0:
                self.crop_box_deleted.emit(index)
            return

        if self.mode == MODE_POINTER and event.button() == Qt.RightButton:
            index = self.pointer_at(self.clamp_point(event.pos()))
            if index >= 0:
                self.pointer_deleted.emit(index)
            return

        if event.button() != Qt.LeftButton:
            return

        if self.mode == MODE_POINTER:
            self.pointer_point_clicked.emit(self.clamp_point(event.pos()))
            return

        self.start_point = self.clamp_point(event.pos())
        self.end_point = self.start_point
        self.selection_rect = QRect(self.start_point, self.end_point).normalized()
        self.selection_changed.emit()
        self.update()

    def mouseMoveEvent(self, event):
        if self.start_point is None or not (event.buttons() & Qt.LeftButton):
            return

        self.end_point = self.clamp_point(event.pos())
        self.selection_rect = QRect(self.start_point, self.end_point).normalized()
        self.selection_changed.emit()
        self.update()

    def mouseReleaseEvent(self, event):
        if self.start_point is None or event.button() != Qt.LeftButton:
            return

        self.end_point = self.clamp_point(event.pos())
        self.selection_rect = QRect(self.start_point, self.end_point).normalized()
        rect = self.selection_rect

        if self.mode == MODE_DETECT:
            if rect.width() > 3 and rect.height() > 3:
                self.detection_box_created.emit(rect)
                self.clear_selection()
            else:
                index = self.box_at(self.end_point)
                self.selected_box_index = index
                self.detection_box_selected.emit(index)
                self.clear_selection()
            return

        if self.mode == MODE_BUSINESS:
            if rect.width() > 3 and rect.height() > 3:
                self.business_box_created.emit(rect)
                self.clear_selection()
            else:
                self.clear_selection()
            return

        if self.mode == MODE_POINTER:
            self.clear_selection()
            return

        if rect.width() > 3 and rect.height() > 3:
            self.crop_box_created.emit(rect)
            self.clear_selection()
            return

        self.selection_changed.emit()
        self.update()

    def color_for_class(self, class_id):
        palette = [
            QColor(14, 165, 233),
            QColor(34, 197, 94),
            QColor(249, 115, 22),
            QColor(239, 68, 68),
            QColor(168, 85, 247),
            QColor(20, 184, 166),
            QColor(234, 179, 8),
            QColor(244, 63, 94),
        ]
        return palette[class_id % len(palette)]

    def color_for_text(self, text):
        value = 0
        for char in str(text):
            value = (value * 131 + ord(char)) & 0xFFFFFFFF
        return self.color_for_class(value)

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self.mode == MODE_DETECT:
            self.paint_connections(painter)
            self.paint_detection_boxes(painter)
        elif self.mode == MODE_CROP:
            self.paint_crop_boxes(painter)
        elif self.mode == MODE_POINTER:
            self.paint_pointer_records(painter)
            self.paint_pointer_draft(painter)
        elif self.mode == MODE_BUSINESS:
            self.paint_business_boxes(painter)

        if self.has_selection():
            selection_rect = self.image_rect_to_view_rect(self.selection_rect)
            painter.fillRect(selection_rect, QColor(14, 165, 233, 48))
            painter.setPen(QPen(QColor(14, 165, 233), 2))
            painter.drawRect(selection_rect)
            if self.mode == MODE_CROP and self.crop_class_label:
                self.paint_selection_label(painter, selection_rect)

    def paint_business_boxes(self, painter):
        for box in self.business_boxes:
            text = box.get("component_type") or ""
            biz_name = box.get("biz_name") or ""
            label = text if not biz_name else f"{text} | {biz_name}"
            color = self.color_for_text(text)
            painter.setPen(QPen(color, 2))
            rect = self.image_to_view_rect(box)
            painter.drawRect(rect)
            self.paint_box_label(painter, rect, label, color)

    def paint_selection_label(self, painter, view_rect):
        label = self.crop_class_label
        metrics = painter.fontMetrics()
        label_w = metrics.horizontalAdvance(label) + 12
        label_h = metrics.height() + 6
        label_y = view_rect.top() - label_h if view_rect.top() >= label_h else view_rect.top()
        label_rect = QRect(view_rect.left(), label_y, label_w, label_h)
        painter.fillRect(label_rect, QColor(14, 165, 233, 225))
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.drawText(label_rect, Qt.AlignCenter, label)

    def paint_crop_boxes(self, painter):
        for box in self.crop_history_boxes:
            color = self.color_for_text(box.get("class_path") or box.get("class_leaf") or "")
            painter.setPen(QPen(color, 2, Qt.DashLine))
            rect = self.image_to_view_rect(box)
            painter.drawRect(rect)
            self.paint_box_label(
                painter,
                rect,
                box.get("class_leaf") or box.get("class_path") or "",
                color,
                prefix="历史 ",
            )

        for box in self.crop_boxes:
            color = self.color_for_text(box.get("class_path") or box.get("class_leaf") or "")
            painter.setPen(QPen(color, 2))
            rect = self.image_to_view_rect(box)
            painter.drawRect(rect)
            self.paint_box_label(painter, rect, box.get("class_leaf") or box.get("class_path") or "", color)

    def paint_box_label(self, painter, rect, label, color=None, prefix=""):
        if not label:
            return
        color = color or QColor(14, 165, 233)
        label = f"{prefix}{label}"
        metrics = painter.fontMetrics()
        label_w = metrics.horizontalAdvance(label) + 12
        label_h = metrics.height() + 6
        label_y = rect.top() - label_h if rect.top() >= label_h else rect.top()
        label_rect = QRect(rect.left(), label_y, label_w, label_h)
        painter.fillRect(label_rect, QColor(color.red(), color.green(), color.blue(), 225))
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.drawText(label_rect, Qt.AlignCenter, label)

    def paint_connections(self, painter):
        if not self.show_connections:
            return

        for source_index, target_index, distance, too_far in self.connections:
            if source_index >= len(self.det_boxes) or target_index >= len(self.det_boxes):
                continue
            source = self.det_boxes[source_index]
            target = self.det_boxes[target_index]
            sx = self.image_to_view_value((source["x1"] + source["x2"]) / 2)
            sy = self.image_to_view_value((source["y1"] + source["y2"]) / 2)
            tx = self.image_to_view_value((target["x1"] + target["x2"]) / 2)
            ty = self.image_to_view_value((target["y1"] + target["y2"]) / 2)
            color = QColor(239, 68, 68) if too_far else QColor(22, 163, 74)
            painter.setPen(QPen(color, 2, Qt.DashLine if too_far else Qt.SolidLine))
            painter.drawLine(sx, sy, tx, ty)
            mid_x = int((sx + tx) / 2)
            mid_y = int((sy + ty) / 2)
            painter.fillRect(QRect(mid_x - 22, mid_y - 10, 44, 20), QColor(17, 24, 39, 180))
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.drawText(QRect(mid_x - 22, mid_y - 10, 44, 20), Qt.AlignCenter, f"{distance:.0f}")

    def paint_detection_boxes(self, painter):
        for index, box in enumerate(self.det_boxes):
            color = self.color_for_class(box["class_id"])
            pen_width = 3 if index == self.selected_box_index else 2
            painter.setPen(QPen(color, pen_width))
            rect = self.image_to_view_rect(box)
            painter.drawRect(rect)

            label = box["class_name"]
            if box.get("score") is not None:
                label = f"{label} {box['score']:.2f}"
            metrics = painter.fontMetrics()
            label_w = metrics.horizontalAdvance(label) + 10
            label_h = metrics.height() + 6
            label_y = rect.top() - label_h if rect.top() >= label_h else rect.top()
            label_rect = QRect(rect.left(), label_y, label_w, label_h)
            painter.fillRect(label_rect, QColor(color.red(), color.green(), color.blue(), 220))
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.drawText(label_rect, Qt.AlignCenter, label)

    def paint_pointer_records(self, painter):
        for index, record in enumerate(self.pointer_records):
            center = record.get("center_xy") or []
            tip = record.get("tip_xy") or []
            bbox = record.get("bbox_xyxy") or []
            if len(center) != 2 or len(tip) != 2:
                continue
            angle = record.get("angle_deg", 0)
            color = self.color_for_class(index + 2)
            if len(bbox) == 4:
                rect_box = {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]}
                painter.setPen(QPen(color, 2, Qt.DashLine))
                painter.drawRect(self.image_to_view_rect(rect_box))
            self.paint_pointer_line(
                painter,
                center[0],
                center[1],
                tip[0],
                tip[1],
                color,
                f"{record.get('label') or record.get('text') or 'knob'} {float(angle):.1f}°",
            )

    def paint_pointer_draft(self, painter):
        if self.pointer_draft_center is None:
            return
        cx = self.image_to_view_value(self.pointer_draft_center.x())
        cy = self.image_to_view_value(self.pointer_draft_center.y())
        color = QColor(14, 165, 233)
        painter.setPen(QPen(color, 2))
        painter.setBrush(color)
        painter.drawEllipse(QPoint(cx, cy), 6, 6)
        label = "中心"
        metrics = painter.fontMetrics()
        label_rect = QRect(cx + 8, cy - metrics.height() - 6, metrics.horizontalAdvance(label) + 12, metrics.height() + 6)
        painter.fillRect(label_rect, QColor(color.red(), color.green(), color.blue(), 225))
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.drawText(label_rect, Qt.AlignCenter, label)

    def paint_pointer_line(self, painter, cx, cy, tx, ty, color, label):
        vcx = self.image_to_view_value(cx)
        vcy = self.image_to_view_value(cy)
        vtx = self.image_to_view_value(tx)
        vty = self.image_to_view_value(ty)
        painter.setPen(QPen(color, 3))
        painter.drawLine(vcx, vcy, vtx, vty)
        painter.setBrush(color)
        painter.drawEllipse(QPoint(vcx, vcy), 5, 5)
        painter.drawEllipse(QPoint(vtx, vty), 4, 4)

        metrics = painter.fontMetrics()
        label_w = metrics.horizontalAdvance(label) + 12
        label_h = metrics.height() + 6
        label_rect = QRect(vtx + 6, vty - label_h - 6, label_w, label_h)
        painter.fillRect(label_rect, QColor(color.red(), color.green(), color.blue(), 225))
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.drawText(label_rect, Qt.AlignCenter, label)
