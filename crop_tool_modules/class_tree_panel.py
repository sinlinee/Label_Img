import json
import os

from PyQt5.QtCore import QModelIndex, Qt, pyqtSignal
from PyQt5.QtWidgets import QAbstractItemView, QGridLayout, QGroupBox, QPushButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout

from .common_utils import clone_class_tree
from .constants import CLASS_TREE_PATH, DEFAULT_CLASS_TREE


def load_class_tree_config(config_path=CLASS_TREE_PATH, default_tree=None):
    tree_data = clone_class_tree(DEFAULT_CLASS_TREE if default_tree is None else default_tree)
    if not os.path.exists(config_path):
        return tree_data

    try:
        with open(config_path, "r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
    except (OSError, json.JSONDecodeError):
        return tree_data

    if isinstance(data, list) and data:
        return clone_class_tree(data)
    return tree_data


def save_class_tree_config(tree_data, config_path=CLASS_TREE_PATH):
    with open(config_path, "w", encoding="utf-8") as file_obj:
        json.dump(tree_data, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")


class ClassTreePanel(QGroupBox):
    class_path_changed = pyqtSignal(str)
    status_message = pyqtSignal(str)
    tree_saved = pyqtSignal(list)

    def __init__(self, tree_data, current_path=None, title="分类树", config_path=CLASS_TREE_PATH):
        super().__init__(title)
        self.tree_data = clone_class_tree(tree_data)
        self.current_path = current_path
        self.config_path = config_path
        self.build_ui()
        self.rebuild_tree()

    def build_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(8)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setMinimumHeight(330)
        self.tree.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.tree.currentItemChanged.connect(self.on_current_item_changed)
        self.tree.itemChanged.connect(self.on_item_changed)
        layout.addWidget(self.tree)

        ops = QGridLayout()
        buttons = [
            ("新增根类", self.add_root),
            ("新增子类", self.add_child),
            ("新增同级", self.add_sibling),
            ("重命名", self.rename_item),
            ("删除", self.delete_item),
            ("保存结构", self.save_tree),
        ]
        for idx, (label, callback) in enumerate(buttons):
            button = QPushButton(label)
            button.clicked.connect(callback)
            ops.addWidget(button, idx // 2, idx % 2)
        layout.addLayout(ops)
        self.setLayout(layout)

    def add_nodes(self, parent, nodes, prefix=None):
        prefix = prefix or []
        for node in nodes:
            name = str(node.get("name", "未命名")).strip() or "未命名"
            item = self.make_item(name)
            item.setData(0, Qt.UserRole, prefix + [name])
            if parent is None:
                self.tree.addTopLevelItem(item)
            else:
                parent.addChild(item)
            self.add_nodes(item, node.get("children", []), prefix + [name])

    def rebuild_tree(self):
        self.tree.clear()
        self.add_nodes(None, self.tree_data)
        self.tree.expandAll()
        self.select_path(self.current_path)

    def data(self):
        def item_to_node(item):
            return {
                "name": item.text(0),
                "children": [item_to_node(item.child(i)) for i in range(item.childCount())],
            }

        return [item_to_node(self.tree.topLevelItem(i)) for i in range(self.tree.topLevelItemCount())]

    def path_from_item(self, item):
        if item is None:
            return ""
        path = []
        while item is not None:
            path.append(item.text(0))
            item = item.parent()
        return "/".join(reversed(path))

    def select_path(self, class_path):
        if not class_path:
            return

        parent = None
        item = None
        for part in class_path.split("/"):
            item = self.find_child(parent, part)
            if item is None:
                return
            parent = item
        self.tree.setCurrentItem(item)

    def clear_selection(self):
        self.tree.blockSignals(True)
        self.tree.clearSelection()
        self.tree.setCurrentIndex(QModelIndex())
        self.current_path = ""
        self.tree.blockSignals(False)
        self.class_path_changed.emit("")

    def find_child(self, parent, name):
        count = self.tree.topLevelItemCount() if parent is None else parent.childCount()
        for index in range(count):
            item = self.tree.topLevelItem(index) if parent is None else parent.child(index)
            if item.text(0) == name:
                return item
        return None

    def selected_item(self):
        return self.tree.currentItem()

    def on_current_item_changed(self, current, _previous):
        self.current_path = self.path_from_item(current)
        self.class_path_changed.emit(self.current_path)

    def on_item_changed(self, item, _column):
        name = item.text(0).strip()
        if not name:
            item.setText(0, "未命名")
            return
        self.current_path = self.path_from_item(item)
        self.class_path_changed.emit(self.current_path)
        self.save_tree()

    def make_item(self, name):
        item = QTreeWidgetItem([name])
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        return item

    def edit_item(self, item):
        self.tree.setCurrentItem(item)
        self.tree.editItem(item, 0)

    def add_root(self):
        item = self.make_item("新分类")
        self.tree.addTopLevelItem(item)
        self.edit_item(item)
        self.save_tree()

    def add_child(self):
        parent = self.selected_item()
        if parent is None:
            self.add_root()
            return
        item = self.make_item("新子类")
        parent.addChild(item)
        parent.setExpanded(True)
        self.edit_item(item)
        self.save_tree()

    def add_sibling(self):
        current = self.selected_item()
        if current is None or current.parent() is None:
            self.add_root()
            return
        parent = current.parent()
        item = self.make_item("新同级")
        parent.addChild(item)
        parent.setExpanded(True)
        self.edit_item(item)
        self.save_tree()

    def rename_item(self):
        item = self.selected_item()
        if item is None:
            return
        self.edit_item(item)

    def delete_item(self):
        item = self.selected_item()
        if item is None:
            return
        parent = item.parent()
        if parent is None:
            index = self.tree.indexOfTopLevelItem(item)
            self.tree.takeTopLevelItem(index)
        else:
            parent.removeChild(item)
        self.current_path = ""
        self.class_path_changed.emit("")
        self.save_tree()

    def save_tree(self):
        self.tree_data = self.data()
        try:
            save_class_tree_config(self.tree_data, self.config_path)
        except OSError as exc:
            self.status_message.emit(f"状态：保存分类树失败：{exc}")
            return
        self.tree_saved.emit(self.tree_data)
        self.status_message.emit(f"状态：已保存树结构 {self.config_path}")
