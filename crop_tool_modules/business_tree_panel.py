from PyQt5.QtCore import QModelIndex, Qt, pyqtSignal
from PyQt5.QtWidgets import QAbstractItemView, QGridLayout, QGroupBox, QPushButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout

from .business_config import normalize_business_tree, save_business_tree_config
from .constants import BUSINESS_TREE_PATH


class BusinessTreePanel(QGroupBox):
    business_path_changed = pyqtSignal(str, str, str)
    status_message = pyqtSignal(str)
    tree_saved = pyqtSignal(list)

    def __init__(self, tree_data, current_path=None):
        super().__init__("业务树")
        self.tree_data = normalize_business_tree(tree_data)
        self.current_path = current_path or ""
        self.build_ui()
        self.rebuild_tree()

    def build_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(8)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["名称", "注释"])
        self.tree.setMinimumHeight(330)
        self.tree.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.tree.currentItemChanged.connect(self.on_current_item_changed)
        self.tree.itemChanged.connect(self.on_item_changed)
        layout.addWidget(self.tree)

        ops = QGridLayout()
        buttons = [
            ("新增组件类型", self.add_root),
            ("新增业务名", self.add_child),
            ("新增同级", self.add_sibling),
            ("重命名/注释", self.rename_item),
            ("删除", self.delete_item),
            ("保存业务树", self.save_tree),
        ]
        for idx, (label, callback) in enumerate(buttons):
            button = QPushButton(label)
            button.clicked.connect(callback)
            ops.addWidget(button, idx // 2, idx % 2)
        layout.addLayout(ops)
        self.setLayout(layout)

    def make_item(self, name, note=""):
        item = QTreeWidgetItem([name, note])
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        return item

    def add_nodes(self, parent, nodes):
        for node in nodes:
            item = self.make_item(node.get("name", ""), node.get("note", ""))
            if parent is None:
                self.tree.addTopLevelItem(item)
            else:
                parent.addChild(item)
            self.add_nodes(item, node.get("children", []))

    def rebuild_tree(self):
        self.tree.clear()
        self.add_nodes(None, self.tree_data)
        self.tree.expandAll()
        self.tree.resizeColumnToContents(0)
        self.select_path(self.current_path)

    def data(self):
        def item_to_node(item):
            return {
                "name": item.text(0),
                "note": item.text(1),
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

    def selected_item(self):
        return self.tree.currentItem()

    def selected_values(self):
        item = self.selected_item()
        if item is None:
            return "", "", ""
        path = self.path_from_item(item)
        parts = [part for part in path.split("/") if part]
        component_type = parts[0] if parts else ""
        biz_name = parts[-1] if len(parts) >= 2 else ""
        note = item.text(1)
        return component_type, biz_name, note

    def find_child(self, parent, name):
        count = self.tree.topLevelItemCount() if parent is None else parent.childCount()
        for index in range(count):
            item = self.tree.topLevelItem(index) if parent is None else parent.child(index)
            if item.text(0) == name:
                return item
        return None

    def select_path(self, path):
        if not path:
            return
        parent = None
        item = None
        for part in path.split("/"):
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
        self.business_path_changed.emit("", "", "")

    def on_current_item_changed(self, current, _previous):
        self.current_path = self.path_from_item(current)
        component_type, biz_name, note = self.selected_values()
        self.business_path_changed.emit(component_type, biz_name, note)

    def on_item_changed(self, item, _column):
        if not item.text(0).strip():
            item.setText(0, "未命名")
        self.current_path = self.path_from_item(item)
        component_type, biz_name, note = self.selected_values()
        self.business_path_changed.emit(component_type, biz_name, note)
        self.save_tree()

    def edit_item(self, item, column=0):
        self.tree.setCurrentItem(item)
        self.tree.editItem(item, column)

    def add_root(self):
        item = self.make_item("component_type", "组件类型说明")
        self.tree.addTopLevelItem(item)
        self.edit_item(item, 0)
        self.save_tree()

    def add_child(self):
        parent = self.selected_item()
        if parent is None:
            self.add_root()
            return
        item = self.make_item("业务名", "中文说明或标注提示")
        parent.addChild(item)
        parent.setExpanded(True)
        self.edit_item(item, 0)
        self.save_tree()

    def add_sibling(self):
        current = self.selected_item()
        if current is None or current.parent() is None:
            self.add_root()
            return
        parent = current.parent()
        item = self.make_item("业务名", "中文说明或标注提示")
        parent.addChild(item)
        parent.setExpanded(True)
        self.edit_item(item, 0)
        self.save_tree()

    def rename_item(self):
        item = self.selected_item()
        if item is None:
            return
        self.edit_item(item, 0)

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
        self.business_path_changed.emit("", "", "")
        self.save_tree()

    def save_tree(self):
        self.tree_data = self.data()
        try:
            self.tree_data = save_business_tree_config(self.tree_data)
        except OSError as exc:
            self.status_message.emit(f"状态：保存业务树失败：{exc}")
            return
        self.tree_saved.emit(self.tree_data)
        self.status_message.emit(f"状态：已保存业务树 {BUSINESS_TREE_PATH}")
