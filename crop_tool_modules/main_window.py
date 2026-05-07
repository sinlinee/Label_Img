import os
import shutil
import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QDoubleSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from crop_tool_modules.common_utils import clamp
from crop_tool_modules.classification_dataset import (
    delete_manifest_record,
    flat_class_name,
    load_manifest,
    save_crop,
    update_manifest_record,
    update_manifest_record_class,
)
from crop_tool_modules.app_settings import load_settings, normalize_user_path, save_settings
from crop_tool_modules.business_config import load_business_tree_config
from crop_tool_modules.business_dataset import (
    add_business_record,
    delete_business_record,
    load_business_manifest,
    update_business_record,
)
from crop_tool_modules.business_tree_panel import BusinessTreePanel
from crop_tool_modules.class_tree_panel import ClassTreePanel, load_class_tree_config
from crop_tool_modules.constants import (
    DEFAULT_INPUT_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_DIR,
    IMAGE_EXTENSIONS,
    MODE_CROP,
    MODE_DETECT,
    MODE_BUSINESS,
    POINTER_GAUGE_TREE_PATH,
    MODE_POINTER,
    POINTER_TREE_PATH,
    REFER_DIR,
)
from crop_tool_modules.detection_mixin import DetectionMixin
from crop_tool_modules.detection_panel import build_detection_panel
from crop_tool_modules.image_io import read_image
from crop_tool_modules.image_label import ImageLabel
from crop_tool_modules.pointer_dataset import (
    angle_from_points,
    delete_pointer_record,
    load_pointer_manifest,
    save_pointer_record,
    update_pointer_record_metadata,
)
from crop_tool_modules.pointer_config import (
    load_pointer_gauge_tree_config,
    load_pointer_object_types,
    load_pointer_tree_config,
    save_pointer_object_types,
)
from crop_tool_modules.stylesheet import APP_STYLESHEET


class Tool(DetectionMixin, QWidget):
    def __init__(self):
        super().__init__()

        self.input_dir = DEFAULT_INPUT_DIR
        self.output_dir = DEFAULT_OUTPUT_DIR
        self.img_list = []
        self.index = 0
        self.img = None
        self.current_image_rel = None
        self.saved_image_rel = None
        self.class_tree_data = []
        self.current_class_path = None
        self.pointer_tree_data = []
        self.pointer_gauge_tree_data = []
        self.current_pointer_path = None
        self.current_pointer_gauge_path = None
        self.pointer_object_types = []
        self.crop_boxes = []
        self.crop_history_records = []
        self.pointer_records = []
        self.business_records = []
        self.business_tree_data = []
        self.current_business_component_type = ""
        self.current_business_biz_name = ""
        self.current_business_note = ""
        self.pointer_draft_center = None
        self.pointer_draft_tip = None
        self.classification_full_image = False
        self.mode = MODE_CROP
        self.sort_image_names = False
        self.recursive_scan = False
        self.zoom_percent = 100

        self.det_class_names = ["knob", "plate", "pointer", "button"]
        self.active_det_class_id = 0
        self.saved_det_class_name = None
        self.det_boxes = []
        self.det_cache = {}
        self.det_dirty = False
        self.labels_index = {}
        self.labels_index_loaded = False
        self.yolo_model = None
        self.yolo_model_names = {}
        self.model_path = DEFAULT_MODEL_PATH if os.path.exists(DEFAULT_MODEL_PATH) else ""
        self.yolo_review_dir = ""

        self.load_app_settings()
        self.class_tree_data = load_class_tree_config()
        self.pointer_tree_data = load_pointer_tree_config()
        self.pointer_gauge_tree_data = load_pointer_gauge_tree_config()
        self.pointer_object_types = load_pointer_object_types()
        self.business_tree_data = load_business_tree_config()
        self.load_detection_classes_config()
        self.setWindowTitle("识别 / 分类 / 角度标注工作台")
        self.resize(1760, 960)
        self.setStyleSheet(APP_STYLESHEET)

        self.build_ui()
        self.reload_image_list()
        self.load_image()

    def build_ui(self):
        self.image_label = ImageLabel()
        self.image_label.selection_changed.connect(self.update_crop_preview)
        self.image_label.crop_box_created.connect(self.add_crop_box)
        self.image_label.crop_box_deleted.connect(self.delete_crop_box)
        self.image_label.detection_box_created.connect(self.add_detection_box)
        self.image_label.detection_box_deleted.connect(self.delete_detection_box)
        self.image_label.detection_box_selected.connect(self.select_detection_box)
        self.image_label.business_box_created.connect(self.add_business_box)
        self.image_label.business_box_deleted.connect(self.delete_business_record_by_index)
        self.image_label.pointer_point_clicked.connect(self.handle_pointer_point_clicked)
        self.image_label.pointer_deleted.connect(self.delete_pointer_record_by_index)

        self.image_scroll = QScrollArea()
        self.image_scroll.setWidget(self.image_label)
        self.image_scroll.setWidgetResizable(False)
        self.image_scroll.setAlignment(Qt.AlignCenter)
        self.image_scroll.setFrameShape(QFrame.NoFrame)
        self.image_scroll.setMinimumSize(820, 680)
        self.image_scroll.setStyleSheet(
            "QScrollArea { background: #0f172a; border: 1px solid #1f2937; border-radius: 18px; }"
        )

        self.image_info = QLabel()
        self.image_info.setProperty("class", "hint")
        self.selection_info = QLabel("裁剪框：未选择")
        self.selection_info.setProperty("class", "hint")

        self.input_dir_value = QLineEdit()
        self.input_dir_value.setPlaceholderText("输入图片目录，可直接粘贴路径后回车")
        self.input_dir_value.returnPressed.connect(self.apply_directories_from_edits)
        self.output_dir_value = QLineEdit()
        self.output_dir_value.setPlaceholderText("输出目录，可直接粘贴路径后回车；不存在会自动创建")
        self.output_dir_value.returnPressed.connect(self.apply_directories_from_edits)

        self.mode_crop_btn = QPushButton("分类模型")
        self.mode_crop_btn.setCheckable(True)
        self.mode_crop_btn.clicked.connect(lambda: self.set_mode(MODE_CROP))
        self.mode_detect_btn = QPushButton("目标识别模型")
        self.mode_detect_btn.setCheckable(True)
        self.mode_detect_btn.clicked.connect(lambda: self.set_mode(MODE_DETECT))
        self.mode_pointer_btn = QPushButton("角度识别模型")
        self.mode_pointer_btn.setCheckable(True)
        self.mode_pointer_btn.clicked.connect(lambda: self.set_mode(MODE_POINTER))
        self.mode_business_btn = QPushButton("业务标注")
        self.mode_business_btn.setCheckable(True)
        self.mode_business_btn.clicked.connect(lambda: self.set_mode(MODE_BUSINESS))

        mode_box = QGroupBox("工作模式")
        mode_layout = QHBoxLayout()
        mode_layout.setSpacing(8)
        mode_layout.addWidget(self.mode_crop_btn)
        mode_layout.addWidget(self.mode_detect_btn)
        mode_layout.addWidget(self.mode_pointer_btn)
        mode_layout.addWidget(self.mode_business_btn)
        mode_box.setLayout(mode_layout)

        btn_input_dir = QPushButton("浏览")
        btn_input_dir.clicked.connect(self.choose_input_dir)
        btn_output_dir = QPushButton("浏览")
        btn_output_dir.clicked.connect(self.choose_output_dir)
        btn_apply_dirs = QPushButton("应用目录 / 刷新")
        btn_apply_dirs.clicked.connect(self.apply_directories_from_edits)

        self.sort_checkbox = QCheckBox("按文件名排序")
        self.sort_checkbox.setChecked(self.sort_image_names)
        self.sort_checkbox.stateChanged.connect(self.toggle_sort_images)
        self.recursive_checkbox = QCheckBox("递归扫描子目录（支持多级）")
        self.recursive_checkbox.setChecked(self.recursive_scan)
        self.recursive_checkbox.setToolTip("开启后会扫描输入目录下所有层级的图片，并按相对路径保存对应 label。")
        self.recursive_checkbox.stateChanged.connect(self.toggle_recursive_scan)

        dir_box = QGroupBox("数据目录")
        dir_layout = QGridLayout()
        dir_layout.setHorizontalSpacing(8)
        dir_layout.setVerticalSpacing(8)
        dir_layout.addWidget(QLabel("输入目录"), 0, 0)
        dir_layout.addWidget(self.input_dir_value, 0, 1)
        dir_layout.addWidget(btn_input_dir, 0, 2)
        dir_layout.addWidget(QLabel("输出目录"), 1, 0)
        dir_layout.addWidget(self.output_dir_value, 1, 1)
        dir_layout.addWidget(btn_output_dir, 1, 2)
        dir_layout.addWidget(self.sort_checkbox, 2, 0, 1, 2)
        dir_layout.addWidget(self.recursive_checkbox, 3, 0, 1, 2)
        dir_layout.addWidget(btn_apply_dirs, 4, 0, 1, 3)
        dir_box.setLayout(dir_layout)

        self.crop_label = QLabel("拖画左侧原图后，这里显示裁剪预览")
        self.crop_label.setFixedSize(300, 300)
        self.crop_label.setAlignment(Qt.AlignCenter)
        self.crop_label.setStyleSheet(
            "background: #f8fafc; border: 1px dashed #94a3b8; border-radius: 16px; color: #475569;"
        )
        self.full_image_class_checkbox = QCheckBox("整图分类（不裁剪）")
        self.full_image_class_checkbox.setChecked(self.classification_full_image)
        self.full_image_class_checkbox.stateChanged.connect(self.toggle_full_image_classification)

        self.crop_history_list = QListWidget()
        self.crop_history_list.setMinimumHeight(150)
        self.crop_history_list.currentRowChanged.connect(self.select_crop_history_record)
        btn_crop_history_reload = QPushButton("刷新历史")
        btn_crop_history_reload.clicked.connect(self.load_crop_history_records)
        btn_crop_history_reclass = QPushButton("改为当前分类")
        btn_crop_history_reclass.clicked.connect(self.reclassify_selected_crop_history)
        btn_crop_history_replace = QPushButton("用新框替换")
        btn_crop_history_replace.clicked.connect(self.replace_selected_crop_history)
        btn_crop_history_delete = QPushButton("删除历史标注")
        btn_crop_history_delete.clicked.connect(self.delete_selected_crop_history)
        crop_history_ops = QHBoxLayout()
        crop_history_ops.addWidget(btn_crop_history_reload)
        crop_history_ops.addWidget(btn_crop_history_reclass)
        crop_history_ops.addWidget(btn_crop_history_replace)
        crop_history_ops.addWidget(btn_crop_history_delete)
        crop_history_box = QGroupBox("历史裁剪标注")
        crop_history_layout = QVBoxLayout()
        crop_history_layout.addWidget(self.crop_history_list)
        crop_history_layout.addLayout(crop_history_ops)
        crop_history_box.setLayout(crop_history_layout)

        self.current_choice = QLabel("当前分类：未选择分类")
        self.current_choice.setProperty("class", "hint")

        self.guide_label = QLabel(
            "分类模型：裁剪目标图并归类。目标识别模型：只标不同目标的检测框。角度识别模型：标旋钮/指针表的 bbox、中心点、尖端点和语义位置。"
        )
        self.guide_label.setWordWrap(True)
        self.guide_label.setProperty("class", "hint")

        self.status_label = QLabel("状态：等待操作")
        self.status_label.setWordWrap(True)
        self.status_label.setProperty("class", "hint")

        self.btn_save = QPushButton("保存当前裁剪")
        self.btn_save.setMinimumHeight(42)
        self.btn_save.clicked.connect(self.save)

        btn_prev = QPushButton("上一张")
        btn_prev.clicked.connect(self.prev_img)

        btn_next = QPushButton("下一张")
        btn_next.clicked.connect(self.next_img)
        btn_reset_history = QPushButton("重置历史记录")
        btn_reset_history.clicked.connect(self.reset_history)
        btn_clear_dataset = QPushButton("删除所有历史和标注")
        btn_clear_dataset.clicked.connect(self.confirm_clear_dataset)

        nav_box = QGroupBox("浏览与保存")
        nav_layout = QVBoxLayout()
        nav_row = QHBoxLayout()
        nav_row.addWidget(btn_prev)
        nav_row.addWidget(btn_next)
        nav_layout.addWidget(self.btn_save)
        nav_layout.addLayout(nav_row)
        nav_layout.addWidget(btn_reset_history)
        nav_layout.addWidget(btn_clear_dataset)
        nav_box.setLayout(nav_layout)

        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(25, 300)
        self.zoom_slider.setSingleStep(5)
        self.zoom_slider.setPageStep(25)
        self.zoom_slider.setValue(self.zoom_percent)
        self.zoom_slider.valueChanged.connect(self.set_zoom_percent)
        self.zoom_label = QLabel(f"缩放：{self.zoom_percent}%")
        btn_zoom_reset = QPushButton("重置缩放")
        btn_zoom_reset.clicked.connect(lambda: self.zoom_slider.setValue(100))
        zoom_box = QGroupBox("图像缩放")
        zoom_layout = QVBoxLayout()
        zoom_layout.addWidget(self.zoom_label)
        zoom_layout.addWidget(self.zoom_slider)
        zoom_layout.addWidget(btn_zoom_reset)
        zoom_box.setLayout(zoom_layout)

        self.ref_grid = QGridLayout()
        self.ref_grid.setSpacing(12)
        self.build_reference_cards()

        ref_wrap = QWidget()
        ref_wrap.setLayout(self.ref_grid)

        ref_scroll = QScrollArea()
        ref_scroll.setWidget(ref_wrap)
        ref_scroll.setWidgetResizable(True)
        ref_scroll.setFrameShape(QFrame.NoFrame)
        ref_scroll.setStyleSheet("QScrollArea { background: transparent; border: 0; }")

        ref_box = QWidget()
        ref_layout = QVBoxLayout()
        ref_layout.setContentsMargins(4, 4, 4, 4)
        ref_layout.addWidget(ref_scroll)
        ref_box.setLayout(ref_layout)

        self.class_tree_panel = ClassTreePanel(self.class_tree_data, self.current_class_path)
        self.class_tree_panel.class_path_changed.connect(self.set_current_class_path)
        self.class_tree_panel.status_message.connect(self.set_status_message)
        self.class_tree_panel.tree_saved.connect(self.on_class_tree_saved)
        class_box = self.class_tree_panel

        detect_box = build_detection_panel(self)

        self.pointer_history_list = QListWidget()
        self.pointer_history_list.setMinimumHeight(260)
        self.pointer_history_list.currentRowChanged.connect(self.select_pointer_record)
        self.pointer_tree_panel = ClassTreePanel(
            self.pointer_tree_data,
            self.current_pointer_path,
            title="旋钮档位树",
            config_path=POINTER_TREE_PATH,
        )
        self.pointer_tree_panel.class_path_changed.connect(self.set_current_pointer_path)
        self.pointer_tree_panel.status_message.connect(self.set_status_message)
        self.pointer_tree_panel.tree_saved.connect(self.on_pointer_tree_saved)
        self.pointer_gauge_tree_panel = ClassTreePanel(
            self.pointer_gauge_tree_data,
            self.current_pointer_gauge_path,
            title="指针读数表类型树",
            config_path=POINTER_GAUGE_TREE_PATH,
        )
        self.pointer_gauge_tree_panel.class_path_changed.connect(self.set_current_pointer_gauge_path)
        self.pointer_gauge_tree_panel.status_message.connect(self.set_status_message)
        self.pointer_gauge_tree_panel.tree_saved.connect(self.on_pointer_gauge_tree_saved)
        self.pointer_text_edit = QLineEdit()
        self.pointer_text_edit.setPlaceholderText("text 可选")
        self.pointer_type_combo = QComboBox()
        self.pointer_type_combo.setEditable(True)
        self.rebuild_pointer_object_type_combo()
        self.pointer_type_combo.currentIndexChanged.connect(self.update_pointer_type_fields)
        self.pointer_type_combo.lineEdit().editingFinished.connect(self.rename_current_pointer_object_type)
        self.pointer_rule_gauge_checkbox = QCheckBox("读数型对象（生成表盘规则）")
        self.pointer_rule_gauge_checkbox.stateChanged.connect(self.set_current_pointer_object_rule_type)
        btn_pointer_type_add = QPushButton("新增对象种类")
        btn_pointer_type_add.clicked.connect(self.add_pointer_object_type)
        btn_pointer_type_delete = QPushButton("删除对象种类")
        btn_pointer_type_delete.clicked.connect(self.delete_pointer_object_type)
        self.pointer_gauge_value_spin = QDoubleSpinBox()
        self.pointer_gauge_value_spin.setRange(-1000000, 1000000)
        self.pointer_gauge_value_spin.setDecimals(3)
        self.pointer_gauge_value_spin.setValue(0)
        self.pointer_gauge_min_spin = QDoubleSpinBox()
        self.pointer_gauge_min_spin.setRange(-1000000, 1000000)
        self.pointer_gauge_min_spin.setDecimals(3)
        self.pointer_gauge_min_spin.setValue(0)
        self.pointer_gauge_max_spin = QDoubleSpinBox()
        self.pointer_gauge_max_spin.setRange(-1000000, 1000000)
        self.pointer_gauge_max_spin.setDecimals(3)
        self.pointer_gauge_max_spin.setValue(100)
        self.pointer_angle_label = QLabel("当前角度：未标注")
        self.pointer_angle_label.setProperty("class", "hint")
        btn_pointer_reload = QPushButton("刷新指针历史")
        btn_pointer_reload.clicked.connect(self.load_pointer_records)
        btn_pointer_delete = QPushButton("删除指针标注")
        btn_pointer_delete.clicked.connect(self.delete_selected_pointer_record)
        btn_pointer_update = QPushButton("按当前设置更新")
        btn_pointer_update.clicked.connect(self.update_selected_pointer_record)
        btn_pointer_reset_draft = QPushButton("重选关键点")
        btn_pointer_reset_draft.clicked.connect(self.reset_pointer_draft)
        btn_pointer_open_rules = QPushButton("打开规则目录")
        btn_pointer_open_rules.clicked.connect(self.open_pointer_rules_dir)
        pointer_ops = QHBoxLayout()
        pointer_ops.addWidget(btn_pointer_reload)
        pointer_ops.addWidget(btn_pointer_reset_draft)
        pointer_ops.addWidget(btn_pointer_update)
        pointer_ops.addWidget(btn_pointer_delete)
        pointer_ops.addWidget(btn_pointer_open_rules)
        pointer_form = QGridLayout()
        pointer_form.setHorizontalSpacing(8)
        pointer_form.setVerticalSpacing(8)
        pointer_form.addWidget(QLabel("对象种类"), 0, 0)
        pointer_form.addWidget(self.pointer_type_combo, 0, 1)
        pointer_type_ops = QHBoxLayout()
        pointer_type_ops.addWidget(btn_pointer_type_add)
        pointer_type_ops.addWidget(btn_pointer_type_delete)
        pointer_form.addLayout(pointer_type_ops, 1, 1)
        pointer_form.addWidget(self.pointer_rule_gauge_checkbox, 2, 1)
        pointer_form.addWidget(QLabel("text"), 3, 0)
        pointer_form.addWidget(self.pointer_text_edit, 3, 1)
        self.pointer_gauge_value_label = QLabel("当前读数")
        pointer_form.addWidget(self.pointer_gauge_value_label, 4, 0)
        pointer_form.addWidget(self.pointer_gauge_value_spin, 4, 1)
        self.pointer_gauge_min_label = QLabel("最小读数")
        pointer_form.addWidget(self.pointer_gauge_min_label, 5, 0)
        pointer_form.addWidget(self.pointer_gauge_min_spin, 5, 1)
        self.pointer_gauge_max_label = QLabel("最大读数")
        pointer_form.addWidget(self.pointer_gauge_max_label, 6, 0)
        pointer_form.addWidget(self.pointer_gauge_max_spin, 6, 1)
        pointer_box = QWidget()
        pointer_layout = QVBoxLayout()
        pointer_layout.setContentsMargins(4, 4, 4, 4)
        pointer_layout.setSpacing(10)
        pointer_hint = QLabel("角度识别：左键点中心，再点指针尖端；自动生成 bbox、连线、角度、关键点和面板规则。右键线段可删除。")
        pointer_hint.setWordWrap(True)
        pointer_hint.setProperty("class", "hint")
        self.pointer_rules_label = QLabel()
        self.pointer_rules_label.setWordWrap(True)
        self.pointer_rules_label.setProperty("class", "hint")
        pointer_layout.addWidget(pointer_hint)
        pointer_layout.addWidget(self.pointer_tree_panel)
        pointer_layout.addWidget(self.pointer_gauge_tree_panel)
        pointer_layout.addLayout(pointer_form)
        pointer_layout.addWidget(self.pointer_angle_label)
        pointer_layout.addWidget(self.pointer_rules_label)
        pointer_layout.addWidget(self.pointer_history_list)
        pointer_layout.addLayout(pointer_ops)
        pointer_layout.addStretch()
        pointer_box.setLayout(pointer_layout)
        self.update_pointer_type_fields()

        self.business_history_list = QListWidget()
        self.business_history_list.setMinimumHeight(260)
        self.business_history_list.currentRowChanged.connect(self.select_business_record)
        self.business_tree_panel = BusinessTreePanel(self.business_tree_data)
        self.business_tree_panel.business_path_changed.connect(self.set_current_business_path)
        self.business_tree_panel.status_message.connect(self.set_status_message)
        self.business_tree_panel.tree_saved.connect(self.on_business_tree_saved)
        self.business_choice_label = QLabel("当前业务：未选择")
        self.business_choice_label.setProperty("class", "hint")
        self.business_note_label = QLabel("注释：")
        self.business_note_label.setWordWrap(True)
        self.business_note_label.setProperty("class", "hint")
        btn_business_reload = QPushButton("刷新业务历史")
        btn_business_reload.clicked.connect(self.load_business_records)
        btn_business_update = QPushButton("按当前字段更新")
        btn_business_update.clicked.connect(self.update_selected_business_record)
        btn_business_delete = QPushButton("删除业务标注")
        btn_business_delete.clicked.connect(self.delete_selected_business_record)
        business_ops = QHBoxLayout()
        business_ops.addWidget(btn_business_reload)
        business_ops.addWidget(btn_business_update)
        business_ops.addWidget(btn_business_delete)
        business_box = QWidget()
        business_layout = QVBoxLayout()
        business_layout.setContentsMargins(4, 4, 4, 4)
        business_layout.setSpacing(10)
        business_hint = QLabel("业务标注：一个组件一个框；component_type 进入 YOLO 类别，biz_name 只保存到业务清单。")
        business_hint.setWordWrap(True)
        business_hint.setProperty("class", "hint")
        business_layout.addWidget(business_hint)
        business_layout.addWidget(self.business_tree_panel)
        business_layout.addWidget(self.business_choice_label)
        business_layout.addWidget(self.business_note_label)
        business_layout.addWidget(self.business_history_list)
        business_layout.addLayout(business_ops)
        business_layout.addStretch()
        business_box.setLayout(business_layout)

        crop_panel = QWidget()
        crop_layout = QVBoxLayout()
        crop_layout.setContentsMargins(4, 4, 4, 4)
        crop_layout.setSpacing(10)
        crop_layout.addWidget(self.crop_label, alignment=Qt.AlignCenter)
        crop_layout.addWidget(self.full_image_class_checkbox)
        crop_layout.addWidget(class_box)
        crop_layout.addWidget(crop_history_box)
        crop_layout.addStretch()
        crop_panel.setLayout(crop_layout)

        self.mode_tabs = QTabWidget()
        self.mode_tabs.addTab(crop_panel, "分类模型")
        self.mode_tabs.addTab(detect_box, "目标识别")
        self.mode_tabs.addTab(pointer_box, "角度识别")
        self.mode_tabs.addTab(business_box, "业务标注")
        self.mode_tabs.addTab(ref_box, "参考图")
        self.mode_tabs.setMinimumWidth(520)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setMinimumWidth(340)
        sidebar.setMaximumWidth(430)
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setContentsMargins(16, 16, 16, 16)
        sidebar_layout.setSpacing(12)

        title = QLabel("识别标注工作台")
        title.setObjectName("appTitle")
        subtitle = QLabel("目标识别、分类识别、角度识别三类数据分开输出")
        subtitle.setWordWrap(True)
        subtitle.setProperty("class", "hint")
        sidebar_layout.addWidget(title)
        sidebar_layout.addWidget(subtitle)
        sidebar_layout.addWidget(mode_box)
        sidebar_layout.addWidget(dir_box)
        sidebar_layout.addWidget(nav_box)
        sidebar_layout.addWidget(zoom_box)
        sidebar_layout.addWidget(self.current_choice)
        sidebar_layout.addWidget(self.guide_label)
        sidebar_layout.addStretch()
        sidebar.setLayout(sidebar_layout)

        canvas_panel = QFrame()
        canvas_panel.setObjectName("canvasPanel")
        canvas_layout = QVBoxLayout()
        canvas_layout.setContentsMargins(16, 16, 16, 16)
        canvas_layout.setSpacing(10)
        canvas_header = QHBoxLayout()
        canvas_title = QLabel("图像画布")
        canvas_title.setObjectName("panelTitle")
        canvas_header.addWidget(canvas_title)
        canvas_header.addStretch()
        canvas_layout.addLayout(canvas_header)
        canvas_layout.addWidget(self.image_scroll, stretch=1)
        canvas_layout.addWidget(self.image_info)
        canvas_layout.addWidget(self.selection_info)
        canvas_layout.addWidget(self.status_label)
        canvas_panel.setLayout(canvas_layout)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(sidebar)
        splitter.addWidget(canvas_panel)
        splitter.addWidget(self.mode_tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([370, 860, 560])

        root = QVBoxLayout()
        root.setContentsMargins(12, 12, 12, 12)
        root.addWidget(splitter)
        self.setLayout(root)
        self.sync_choice_controls()
        self.set_mode(self.mode)

    def build_reference_cards(self):
        def collect_leaves(nodes, prefix=None):
            prefix = prefix or []
            leaves = []
            for node in nodes:
                path = prefix + [node.get("name", "未命名")]
                children = node.get("children", [])
                if children:
                    leaves.extend(collect_leaves(children, path))
                else:
                    leaves.append(path)
            return leaves

        for idx, path in enumerate(collect_leaves(self.class_tree_data)):
            card = QFrame()
            card.setStyleSheet(
                "QFrame { background: #ffffff; border: 1px solid #d1d5db; border-radius: 12px; }"
            )
            card_layout = QVBoxLayout()
            card_layout.setContentsMargins(10, 10, 10, 10)
            card_layout.setSpacing(8)

            image_widget = QLabel()
            image_widget.setAlignment(Qt.AlignCenter)
            image_widget.setFixedSize(150, 150)
            image_widget.setStyleSheet(
                "background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;"
            )

            ref_path = os.path.join(REFER_DIR, f"{idx + 1}.jpg")
            if os.path.exists(ref_path):
                pixmap = QPixmap(ref_path)
                image_widget.setPixmap(
                    pixmap.scaled(140, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
            else:
                image_widget.setText("未找到参考图")

            title = QLabel(path[-1])
            title.setAlignment(Qt.AlignCenter)
            title.setProperty("class", "cardTitle")

            code_text = QLabel(" / ".join(path[:-1]))
            code_text.setAlignment(Qt.AlignCenter)
            code_text.setProperty("class", "hint")

            card_layout.addWidget(image_widget)
            card_layout.addWidget(title)
            card_layout.addWidget(code_text)
            card.setLayout(card_layout)
            self.ref_grid.addWidget(card, idx // 2, idx % 2)

    def update_directory_labels(self):
        self.input_dir_value.setText(self.input_dir)
        self.output_dir_value.setText(self.output_dir)
        if hasattr(self, "pointer_rules_label"):
            self.pointer_rules_label.setText(
                f"面板规则：{os.path.join(self.output_dir, 'pointer_panel_rules.json')}"
            )

    def set_zoom_percent(self, value):
        self.zoom_percent = int(value)
        self.zoom_label.setText(f"缩放：{self.zoom_percent}%")
        if hasattr(self, "image_label"):
            self.image_label.set_zoom_factor(self.zoom_percent / 100)
        self.update_crop_preview()

    def toggle_full_image_classification(self):
        self.classification_full_image = self.full_image_class_checkbox.isChecked()
        self.crop_boxes = []
        self.image_label.set_crop_boxes(self.crop_boxes)
        self.image_label.clear_selection()
        self.update_crop_preview()
        if self.classification_full_image:
            self.set_status_message("状态：已切换为整图分类，不需要拖裁剪框。")
        else:
            self.set_status_message("状态：已切换为裁剪分类，请在图上拖框。")

    def load_app_settings(self):
        settings = load_settings()
        if not settings:
            return

        input_dir = normalize_user_path(settings.get("input_dir", ""))
        output_dir = normalize_user_path(settings.get("output_dir", ""))
        if input_dir:
            self.input_dir = input_dir
        if output_dir:
            self.output_dir = output_dir

        self.sort_image_names = bool(settings.get("sort_image_names", self.sort_image_names))
        self.recursive_scan = bool(settings.get("recursive_scan", self.recursive_scan))
        self.mode = settings.get("mode", self.mode)
        if self.mode not in (MODE_CROP, MODE_DETECT, MODE_POINTER, MODE_BUSINESS):
            self.mode = MODE_CROP

        self.current_class_path = settings.get("current_class_path") or None
        self.current_pointer_path = settings.get("current_pointer_path") or None
        self.current_pointer_gauge_path = settings.get("current_pointer_gauge_path") or None
        try:
            self.active_det_class_id = int(settings.get("active_det_class_id", self.active_det_class_id))
        except (TypeError, ValueError):
            self.active_det_class_id = 0
        self.saved_det_class_name = settings.get("active_det_class_name")
        self.saved_image_rel = settings.get("current_image_rel") or None
        self.yolo_review_dir = normalize_user_path(settings.get("yolo_review_dir", "")) or self.yolo_review_dir

    def save_app_settings(self):
        active_det_class_name = ""
        if 0 <= self.active_det_class_id < len(self.det_class_names):
            active_det_class_name = self.det_class_names[self.active_det_class_id]

        settings = {
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "sort_image_names": self.sort_image_names,
            "recursive_scan": self.recursive_scan,
            "mode": self.mode,
            "current_class_path": self.current_class_path,
            "current_pointer_path": self.current_pointer_path,
            "current_pointer_gauge_path": self.current_pointer_gauge_path,
            "active_det_class_id": self.active_det_class_id,
            "active_det_class_name": active_det_class_name,
            "current_image_rel": self.current_image_rel or self.saved_image_rel,
            "yolo_review_dir": self.yolo_review_dir,
        }
        try:
            save_settings(settings)
        except OSError as exc:
            self.set_status_message(f"Status: failed to save directory settings: {exc}")

    def reset_history(self):
        if not self.save_dirty_detection_if_needed():
            return

        self.index = 0
        self.current_image_rel = None
        self.saved_image_rel = None
        self.current_class_path = None
        self.current_pointer_path = None
        self.current_pointer_gauge_path = None
        self.active_det_class_id = 0
        self.saved_det_class_name = None
        self.crop_boxes = []
        self.det_boxes = []
        self.pointer_records = []
        self.pointer_draft_center = None
        self.pointer_draft_tip = None
        self.det_cache.clear()
        self.det_dirty = False
        self.labels_index = {}
        self.labels_index_loaded = False

        if hasattr(self, "image_label"):
            self.image_label.clear_selection()
            self.image_label.set_crop_boxes(self.crop_boxes)
            self.image_label.set_crop_history_boxes([])
            self.image_label.set_detection_boxes(self.det_boxes)
            self.image_label.set_pointer_records([])
            self.image_label.set_pointer_draft_center(None)

        self.crop_history_records = []
        if hasattr(self, "crop_history_list"):
            self.crop_history_list.clear()
        if hasattr(self, "pointer_history_list"):
            self.pointer_history_list.clear()
        if hasattr(self, "pointer_angle_label"):
            self.pointer_angle_label.setText("当前角度：未标注")

        if hasattr(self, "class_tree_panel"):
            self.class_tree_panel.clear_selection()

        if hasattr(self, "det_class_buttons"):
            for idx, button in enumerate(self.det_class_buttons):
                button.setChecked(idx == self.active_det_class_id)

        self.reload_image_list()
        self.save_app_settings()
        self.load_image()
        self.update_choice_text()
        self.update_crop_class_label()
        self.set_status_message("状态：已重置历史记录，数据集文件和分类树未删除。")

    def confirm_clear_dataset(self):
        output_dir = os.path.abspath(self.output_dir)
        message = (
            "确认删除所有历史记录和标注？\n\n"
            f"将清空输出目录：\n{output_dir}\n\n"
            "这会删除已导出的分类识别数据、目标识别标注、角度识别关键点标注、manifest 和类别文件。"
            "不会删除输入原图、分类树结构和程序设置。"
        )
        reply = QMessageBox.warning(
            self,
            "确认删除数据集",
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.clear_dataset_outputs()

    def clear_dataset_outputs(self):
        output_dir = os.path.abspath(self.output_dir)
        base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        if output_dir in ("", os.path.abspath(os.sep)):
            self.set_status_message("状态：输出目录无效，已取消删除。")
            return
        if output_dir == base_dir:
            self.set_status_message("状态：输出目录指向项目根目录，已取消删除。")
            return

        try:
            if os.path.isdir(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)
        except OSError as exc:
            self.set_status_message(f"状态：删除数据集失败：{exc}")
            return

        self.crop_boxes = []
        self.crop_history_records = []
        self.det_boxes = []
        self.det_cache.clear()
        self.pointer_records = []
        self.pointer_draft_center = None
        self.pointer_draft_tip = None
        self.det_dirty = False
        self.labels_index = {}
        self.labels_index_loaded = False
        self.det_class_names = ["knob", "plate", "pointer", "button"]
        self.active_det_class_id = 0
        self.saved_det_class_name = None
        self.load_detection_classes_config()

        if hasattr(self, "image_label"):
            self.image_label.clear_selection()
            self.image_label.set_crop_boxes(self.crop_boxes)
            self.image_label.set_crop_history_boxes([])
            self.image_label.set_detection_boxes(self.det_boxes)
            self.image_label.set_pointer_records([])
            self.image_label.set_pointer_draft_center(None)
        if hasattr(self, "crop_history_list"):
            self.crop_history_list.clear()
            self.crop_history_list.addItem("当前图片没有历史裁剪标注")
        if hasattr(self, "pointer_history_list"):
            self.pointer_history_list.clear()
            self.pointer_history_list.addItem("当前图片没有角度识别标注")
        if hasattr(self, "pointer_angle_label"):
            self.pointer_angle_label.setText("当前角度：未标注")
        if hasattr(self, "det_classes_edit"):
            self.det_classes_edit.setText(",".join(self.det_class_names))
            self.rebuild_detection_class_buttons()

        self.index = 0
        self.current_image_rel = None
        self.saved_image_rel = None
        self.reload_image_list()
        self.load_image()
        self.update_crop_preview()
        self.update_detection_overlay()
        self.update_choice_text()
        self.save_app_settings()
        self.set_status_message(f"状态：已删除所有历史记录和标注，并重新创建输出目录 {output_dir}。")

    def apply_directories_from_edits(self):
        input_dir = normalize_user_path(self.input_dir_value.text())
        output_dir = normalize_user_path(self.output_dir_value.text())

        if not input_dir or not os.path.isdir(input_dir):
            self.set_status_message(f"状态：输入目录不存在：{input_dir or '(空)'}")
            self.update_directory_labels()
            return

        if not output_dir:
            self.set_status_message("状态：输出目录不能为空。")
            self.update_directory_labels()
            return

        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as exc:
            self.set_status_message(f"状态：无法创建输出目录 {output_dir}：{exc}")
            self.update_directory_labels()
            return

        if not self.save_dirty_detection_if_needed():
            self.update_directory_labels()
            return

        input_changed = input_dir != self.input_dir
        output_changed = output_dir != self.output_dir
        self.input_dir = input_dir
        self.output_dir = output_dir

        if input_changed:
            self.index = 0
            self.saved_image_rel = None
            self.det_cache.clear()
        if output_changed:
            self.labels_index = {}
            self.labels_index_loaded = False
            self.det_cache.clear()
            self.load_detection_classes_config()
            self.det_classes_edit.setText(",".join(self.det_class_names))
            self.rebuild_detection_class_buttons()

        self.reload_image_list()
        self.load_image()
        self.save_app_settings()
        if self.img_list:
            self.set_status_message(
                f"状态：目录已应用，找到 {len(self.img_list)} 张图片。"
            )
        else:
            self.set_status_message(f"状态：目录已应用，但输入目录中没有可处理图片：{self.input_dir}")

    def toggle_sort_images(self):
        self.sort_image_names = self.sort_checkbox.isChecked()
        self.save_app_settings()
        self.reload_image_list()
        self.load_image()

    def toggle_recursive_scan(self):
        self.recursive_scan = self.recursive_checkbox.isChecked()
        self.save_app_settings()
        self.reload_image_list()
        self.load_image()

    def scan_images(self):
        if not os.path.isdir(self.input_dir):
            return []

        images = []
        if not self.recursive_scan:
            with os.scandir(self.input_dir) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name.lower().endswith(IMAGE_EXTENSIONS):
                        images.append(entry.name)
        else:
            stack = [self.input_dir]
            while stack:
                current_dir = stack.pop()
                with os.scandir(current_dir) as entries:
                    for entry in entries:
                        if entry.is_dir():
                            stack.append(entry.path)
                        elif entry.is_file() and entry.name.lower().endswith(IMAGE_EXTENSIONS):
                            images.append(os.path.relpath(entry.path, self.input_dir))

        if self.sort_image_names:
            images.sort()
        return images

    def reload_image_list(self):
        self.update_directory_labels()
        self.img_list = self.scan_images()
        if self.saved_image_rel in self.img_list:
            self.index = self.img_list.index(self.saved_image_rel)
        elif self.current_image_rel in self.img_list:
            self.index = self.img_list.index(self.current_image_rel)
        elif self.img_list:
            self.index = clamp(self.index, 0, len(self.img_list) - 1)
        else:
            self.index = 0

    def clear_image_display(self):
        self.img = None
        self.current_image_rel = None
        self.det_boxes = []
        self.crop_boxes = []
        self.crop_history_records = []
        self.pointer_records = []
        self.pointer_draft_center = None
        self.pointer_draft_tip = None
        self.image_label.clear()
        self.image_label.set_crop_boxes(self.crop_boxes)
        self.image_label.set_crop_history_boxes([])
        self.image_label.set_detection_boxes(self.det_boxes)
        self.image_label.set_pointer_records([])
        self.image_label.set_pointer_draft_center(None)
        self.image_label.clear_selection()
        if hasattr(self, "crop_history_list"):
            self.crop_history_list.clear()
        if hasattr(self, "pointer_history_list"):
            self.pointer_history_list.clear()
        if hasattr(self, "pointer_angle_label"):
            self.pointer_angle_label.setText("当前角度：未标注")
        self.crop_label.setPixmap(QPixmap())
        self.crop_label.setText("拖画左侧原图后，这里显示裁剪预览")
        self.image_info.setText("当前图片：无")
        self.selection_info.setText("裁剪框：未选择")
        self.update_detection_info()

    def choose_input_dir(self):
        selected_dir = QFileDialog.getExistingDirectory(self, "选择输入文件夹", self.input_dir)
        if not selected_dir:
            return

        self.input_dir_value.setText(selected_dir)
        self.apply_directories_from_edits()

    def choose_output_dir(self):
        selected_dir = QFileDialog.getExistingDirectory(self, "选择输出文件夹", self.output_dir)
        if not selected_dir:
            return

        self.output_dir_value.setText(selected_dir)
        self.apply_directories_from_edits()

    def current_image_path(self):
        if not self.current_image_rel:
            return None
        return os.path.join(self.input_dir, self.current_image_rel)

    def load_image(self):
        if not self.img_list:
            self.clear_image_display()
            self.set_status_message(f"状态：在 {self.input_dir} 中没有找到可处理的图片。")
            return

        if self.index >= len(self.img_list):
            self.index = len(self.img_list) - 1
            self.set_status_message("状态：图片已经处理完。")
            return

        self.current_image_rel = self.img_list[self.index]
        self.saved_image_rel = self.current_image_rel
        image_path = self.current_image_path()
        self.img = read_image(image_path)
        if self.img is None:
            self.clear_image_display()
            self.set_status_message(f"状态：无法读取图片 {image_path}")
            return

        rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb.shape
        qimg = QImage(rgb.data, width, height, channel * width, QImage.Format_RGB888).copy()

        self.image_label.set_image(QPixmap.fromImage(qimg))
        self.image_label.set_zoom_factor(self.zoom_percent / 100)
        self.crop_label.setPixmap(QPixmap())
        self.crop_label.setText("拖画左侧原图后，这里显示裁剪预览")
        self.crop_boxes = []
        self.image_label.set_crop_boxes(self.crop_boxes)
        self.load_crop_history_records()
        self.load_pointer_records()
        self.load_business_records()
        self.image_info.setText(
            f"当前图片：{self.current_image_rel}    原始尺寸：{width} x {height}    进度：{self.index + 1}/{len(self.img_list)}"
        )
        self.selection_info.setText("裁剪框：未选择")
        self.load_current_detection_labels()
        self.set_status_message("状态：等待操作")
        self.update_choice_text()
        self.update_crop_class_label()
        self.save_app_settings()

    def prev_img(self):
        if not self.img_list:
            return

        if self.index <= 0:
            self.index = 0
            self.set_status_message("状态：已经是第一张。")
            return

        if not self.save_dirty_detection_if_needed():
            return
        self.index -= 1
        self.load_image()

    def next_img(self):
        if not self.img_list:
            return

        if self.index >= len(self.img_list) - 1:
            self.set_status_message("状态：当前已经是最后一张。")
            return

        if self.mode == MODE_DETECT:
            save_message = "状态：已跳过当前图片。"
            if self.det_dirty or self.det_boxes:
                saved, message = self.save_current_labels()
                if saved:
                    save_message = message
                else:
                    self.set_status_message(message)
                    return
        elif self.mode == MODE_POINTER:
            save_message = "状态：已切换角度识别图片。"
        else:
            save_message = "状态：已跳过当前图片。"
            if self.crop_boxes:
                saved, message = self.save_current_crop()
                if saved:
                    save_message = message
                else:
                    self.set_status_message(message)
                    return

        self.index += 1
        self.load_image()
        self.set_status_message(f"{save_message} 已切换到下一张。")

    def set_mode(self, mode):
        self.mode = mode
        self.mode_crop_btn.setChecked(mode == MODE_CROP)
        self.mode_detect_btn.setChecked(mode == MODE_DETECT)
        self.mode_pointer_btn.setChecked(mode == MODE_POINTER)
        self.mode_business_btn.setChecked(mode == MODE_BUSINESS)
        self.image_label.set_mode(mode)
        if mode == MODE_CROP:
            self.btn_save.setText("保存分类样本")
            self.selection_info.setText("裁剪框：未选择")
            if hasattr(self, "mode_tabs"):
                self.mode_tabs.setCurrentIndex(0)
        elif mode == MODE_DETECT:
            self.btn_save.setText("保存目标识别标注")
            self.selection_info.setText("目标识别：拖框新增目标，右键删除框")
            if hasattr(self, "mode_tabs"):
                self.mode_tabs.setCurrentIndex(1)
        elif mode == MODE_POINTER:
            self.btn_save.setText("保存角度关键点")
            self.selection_info.setText("旋钮对象：先点中心，再点指针尖端")
            if hasattr(self, "mode_tabs"):
                self.mode_tabs.setCurrentIndex(2)
        else:
            self.btn_save.setText("业务标注自动保存")
            self.selection_info.setText("业务标注：拖框新增组件，右键删除框")
            if hasattr(self, "mode_tabs"):
                self.mode_tabs.setCurrentIndex(3)
        self.update_crop_preview()
        self.update_detection_overlay()
        self.update_crop_class_label()
        self.save_app_settings()

    def sync_choice_controls(self):
        self.class_tree_panel.select_path(self.current_class_path)
        self.pointer_tree_panel.select_path(self.current_pointer_path)
        self.pointer_gauge_tree_panel.select_path(self.current_pointer_gauge_path)
        for idx, button in enumerate(self.det_class_buttons):
            button.setChecked(idx == self.active_det_class_id)
        self.update_choice_text()

    def set_current_class_path(self, class_path):
        self.current_class_path = class_path or None
        self.update_choice_text()
        self.update_crop_class_label()
        self.save_app_settings()

    def on_class_tree_saved(self, tree_data):
        self.class_tree_data = tree_data
        self.save_app_settings()

    def set_current_pointer_path(self, pointer_path):
        self.current_pointer_path = pointer_path or None
        self.update_choice_text()
        self.save_app_settings()

    def on_pointer_tree_saved(self, tree_data):
        self.pointer_tree_data = tree_data
        self.save_app_settings()

    def set_current_pointer_gauge_path(self, pointer_path):
        self.current_pointer_gauge_path = pointer_path or None
        self.update_choice_text()
        self.save_app_settings()

    def on_pointer_gauge_tree_saved(self, tree_data):
        self.pointer_gauge_tree_data = tree_data
        self.save_app_settings()

    def update_choice_text(self):
        class_text = self.current_class_path or "未选择分类"
        active_class = self.det_class_names[self.active_det_class_id] if self.det_class_names else "无"
        pointer_text = self.current_pointer_gauge_path if self.current_pointer_rule_type() == "gauge" else self.current_pointer_path
        pointer_text = pointer_text or "未选择角度类型"
        business_text = self.current_business_component_type or ""
        business_text = business_text or "未填写业务类型"
        self.current_choice.setText(
            f"分类类别：{class_text}；识别目标：{active_class}；角度类型：{pointer_text}；业务类型：{business_text}"
        )

    def update_crop_class_label(self):
        if not hasattr(self, "image_label"):
            return
        self.image_label.set_crop_class_label(self.selected_class_leaf())

    def update_pointer_type_fields(self):
        if not hasattr(self, "pointer_type_combo"):
            return
        if not hasattr(self, "pointer_gauge_value_label"):
            return
        self.sync_pointer_rule_checkbox()
        is_gauge = self.current_pointer_rule_type() == "gauge"
        for widget in (
            self.pointer_gauge_value_label,
            self.pointer_gauge_value_spin,
            self.pointer_gauge_min_label,
            self.pointer_gauge_min_spin,
            self.pointer_gauge_max_label,
            self.pointer_gauge_max_spin,
            self.pointer_gauge_tree_panel,
        ):
            widget.setVisible(is_gauge)
        self.pointer_tree_panel.setVisible(not is_gauge)

    def rebuild_pointer_object_type_combo(self, current_name=None):
        if not hasattr(self, "pointer_type_combo"):
            return
        current_name = current_name or self.pointer_type_combo.currentText().strip()
        self.pointer_type_combo.blockSignals(True)
        self.pointer_type_combo.clear()
        for item in self.pointer_object_types:
            name = item.get("name", "")
            self.pointer_type_combo.addItem(name, item.get("rule_type") or name)
        if current_name:
            index = self.pointer_type_combo.findText(current_name)
            if index >= 0:
                self.pointer_type_combo.setCurrentIndex(index)
            else:
                self.pointer_type_combo.setEditText(current_name)
        self.pointer_type_combo.blockSignals(False)
        self.sync_pointer_rule_checkbox()
        self.update_pointer_type_fields()

    def current_pointer_object_name(self):
        return self.pointer_type_combo.currentText().strip()

    def current_pointer_rule_type(self):
        name = self.current_pointer_object_name()
        for item in self.pointer_object_types:
            if item.get("name") == name:
                return item.get("rule_type") or name
        return name

    def sync_pointer_rule_checkbox(self):
        if not hasattr(self, "pointer_rule_gauge_checkbox"):
            return
        self.pointer_rule_gauge_checkbox.blockSignals(True)
        self.pointer_rule_gauge_checkbox.setChecked(self.current_pointer_rule_type() == "gauge")
        self.pointer_rule_gauge_checkbox.blockSignals(False)

    def set_current_pointer_object_rule_type(self):
        name = self.current_pointer_object_name()
        if not name:
            return
        rule_type = "gauge" if self.pointer_rule_gauge_checkbox.isChecked() else "rotary"
        current_index = self.pointer_type_combo.currentIndex()
        if 0 <= current_index < len(self.pointer_object_types):
            self.pointer_object_types[current_index]["name"] = name
            self.pointer_object_types[current_index]["rule_type"] = rule_type
        else:
            self.pointer_object_types.append({"name": name, "rule_type": rule_type})
        if self.save_pointer_object_type_config():
            self.rebuild_pointer_object_type_combo(name)
        self.update_pointer_type_fields()

    def save_pointer_object_type_config(self):
        try:
            self.pointer_object_types = save_pointer_object_types(self.pointer_object_types)
        except OSError as exc:
            self.set_status_message(f"状态：保存角度对象种类失败：{exc}")
            return False
        return True

    def add_pointer_object_type(self):
        base = "新对象种类"
        existing = {item.get("name") for item in self.pointer_object_types}
        name = base
        index = 1
        while name in existing:
            index += 1
            name = f"{base}{index}"
        self.pointer_object_types.append({"name": name, "rule_type": "rotary"})
        if self.save_pointer_object_type_config():
            self.rebuild_pointer_object_type_combo(name)
            self.pointer_type_combo.lineEdit().selectAll()
            self.pointer_type_combo.lineEdit().setFocus()
            self.set_status_message("状态：已新增角度对象种类，直接在下拉框里改名。")

    def delete_pointer_object_type(self):
        name = self.current_pointer_object_name()
        if not name:
            return
        self.pointer_object_types = [item for item in self.pointer_object_types if item.get("name") != name]
        if self.save_pointer_object_type_config():
            self.rebuild_pointer_object_type_combo()
            self.set_status_message(f"状态：已删除角度对象种类 {name}。")

    def rename_current_pointer_object_type(self):
        name = self.current_pointer_object_name()
        if not name:
            return
        current_index = self.pointer_type_combo.currentIndex()
        if 0 <= current_index < len(self.pointer_object_types):
            self.pointer_object_types[current_index]["name"] = name
            self.pointer_object_types[current_index]["rule_type"] = (
                "gauge" if self.pointer_rule_gauge_checkbox.isChecked() else "rotary"
            )
        elif not any(item.get("name") == name for item in self.pointer_object_types):
            self.pointer_object_types.append(
                {"name": name, "rule_type": "gauge" if self.pointer_rule_gauge_checkbox.isChecked() else "rotary"}
            )
        if self.save_pointer_object_type_config():
            self.rebuild_pointer_object_type_combo(name)

    def open_pointer_rules_dir(self):
        os.makedirs(self.output_dir, exist_ok=True)
        rules_path = os.path.join(self.output_dir, "pointer_panel_rules.json")
        if not os.path.exists(rules_path):
            self.set_status_message(f"状态：还没有生成面板规则，点完中心和尖端后会生成：{rules_path}")
        else:
            self.set_status_message(f"状态：面板规则文件：{rules_path}")
        os.startfile(self.output_dir)

    def set_current_business_path(self, component_type, biz_name, note):
        self.current_business_component_type = component_type or ""
        self.current_business_biz_name = biz_name or ""
        self.current_business_note = note or ""
        if hasattr(self, "business_choice_label"):
            if self.current_business_component_type and self.current_business_biz_name:
                self.business_choice_label.setText(
                    f"当前业务：{self.current_business_component_type} / {self.current_business_biz_name}"
                )
            elif self.current_business_component_type:
                self.business_choice_label.setText(f"当前业务：{self.current_business_component_type} / 未选择业务名")
            else:
                self.business_choice_label.setText("当前业务：未选择")
        if hasattr(self, "business_note_label"):
            self.business_note_label.setText(f"注释：{self.current_business_note}")
        self.update_choice_text()

    def on_business_tree_saved(self, tree_data):
        self.business_tree_data = tree_data

    def set_status_message(self, message):
        self.status_label.setText(message)

    def load_pointer_records(self):
        if not hasattr(self, "pointer_history_list"):
            return
        self.pointer_history_list.blockSignals(True)
        self.pointer_history_list.clear()
        self.pointer_records = [
            record for record in load_pointer_manifest(self.output_dir)
            if record.get("image_rel") == self.current_image_rel
        ]
        for index, record in enumerate(self.pointer_records):
            center = record.get("center_xy") or []
            tip = record.get("tip_xy") or []
            bbox = record.get("bbox_xyxy") or []
            angle = float(record.get("angle_deg", 0))
            panel_type = record.get("panel_type") or record.get("panel_id") or ""
            object_type = record.get("object_type") or record.get("type") or "rotary"
            type_name = record.get("object_name") or ("指针表" if object_type == "gauge" else "旋钮")
            name = panel_type if object_type == "gauge" else (record.get("label") or record.get("text") or "未命名")
            gauge_text = ""
            if object_type == "gauge" and record.get("gauge_value") is not None:
                gauge_text = f" value:{record.get('gauge_value')}"
            self.pointer_history_list.addItem(
                f"{index + 1}. {type_name} {name} {angle:.1f}°{gauge_text}  panel_type:{panel_type} bbox:{bbox} center:{center} tip:{tip}"
            )
        if not self.pointer_records:
            self.pointer_history_list.addItem("当前图片没有角度识别标注")
        self.pointer_history_list.blockSignals(False)
        self.image_label.set_pointer_records(self.pointer_records)
        self.update_choice_text()

    def selected_pointer_record(self):
        if not hasattr(self, "pointer_history_list"):
            return None
        row = self.pointer_history_list.currentRow()
        if row < 0 or row >= len(self.pointer_records):
            return None
        return self.pointer_records[row]

    def select_pointer_record(self, row):
        if row < 0 or row >= len(self.pointer_records):
            return
        record = self.pointer_records[row]
        object_type = record.get("object_type") or record.get("type") or "rotary"
        type_name = record.get("object_name") or ("指针表" if object_type == "gauge" else "旋钮")
        label = record.get("label") or ""
        text = record.get("text") or ""
        panel_type = record.get("panel_type") or record.get("panel_id") or ""
        self.pointer_text_edit.setText(text)
        if object_type == "gauge" and panel_type:
            self.pointer_gauge_tree_panel.select_path(panel_type)
        elif panel_type and label:
            self.pointer_tree_panel.select_path(f"{panel_type}/{label}")
        object_name = record.get("object_name") or object_type
        type_index = self.pointer_type_combo.findText(object_name)
        if type_index < 0:
            type_index = self.pointer_type_combo.findData(object_type)
        if type_index >= 0:
            self.pointer_type_combo.setCurrentIndex(type_index)
        else:
            self.pointer_type_combo.setEditText(object_name)
        if record.get("gauge_value") is not None:
            self.pointer_gauge_value_spin.setValue(float(record.get("gauge_value")))
        if record.get("gauge_min_value") is not None:
            self.pointer_gauge_min_spin.setValue(float(record.get("gauge_min_value")))
        if record.get("gauge_max_value") is not None:
            self.pointer_gauge_max_spin.setValue(float(record.get("gauge_max_value")))
        self.selection_info.setText(
            f"{type_name}对象：{panel_type if object_type == 'gauge' else (record.get('label') or record.get('text') or '')} "
            f"{float(record.get('angle_deg', 0)):.1f}°  panel_type:{record.get('panel_type') or record.get('panel_id') or ''} bbox{record.get('bbox_xyxy')} "
            f"center{record.get('center_xy')} tip{record.get('tip_xy')}"
        )
        self.pointer_angle_label.setText(f"当前角度：{float(record.get('angle_deg', 0)):.1f}°")

    def reset_pointer_draft(self):
        self.pointer_draft_center = None
        self.pointer_draft_tip = None
        self.image_label.set_pointer_draft_center(None)
        self.pointer_angle_label.setText("当前角度：未标注")
        self.selection_info.setText("旋钮对象：先点中心，再点指针尖端")

    def handle_pointer_point_clicked(self, point):
        if self.pointer_draft_center is None:
            self.pointer_draft_center = point
            self.pointer_draft_tip = None
            self.image_label.set_pointer_draft_center(point)
            self.pointer_angle_label.setText(f"当前角度：已选择中心 ({point.x()}, {point.y()})")
            self.selection_info.setText("旋钮对象：已点中心，请点指针尖端")
            return

        self.pointer_draft_tip = point
        center = self.pointer_draft_center
        angle = angle_from_points((center.x(), center.y()), (point.x(), point.y()))
        self.pointer_angle_label.setText(f"当前角度：{angle:.1f}°")
        self.add_pointer_record(center, point)
        self.pointer_draft_center = None
        self.pointer_draft_tip = None
        self.image_label.set_pointer_draft_center(None)

    def add_pointer_record(self, center, tip):
        if self.img is None or not self.current_image_rel:
            return
        valid, message = self.validate_selected_pointer_type()
        if not valid:
            self.set_status_message(message)
            return
        label = self.selected_pointer_label()
        text = self.pointer_text_edit.text().strip()
        panel_type = self.selected_pointer_panel_type()
        object_type = self.current_pointer_rule_type()
        saved, result = save_pointer_record(
            self.output_dir,
            self.input_dir,
            self.current_image_rel,
            self.img,
            (center.x(), center.y()),
            (tip.x(), tip.y()),
            label=label,
            text=text,
            panel_type=panel_type,
            object_name=self.current_pointer_object_name(),
            object_type=object_type,
            gauge_value=self.pointer_gauge_value_spin.value() if object_type == "gauge" else None,
            gauge_min_value=self.pointer_gauge_min_spin.value() if object_type == "gauge" else None,
            gauge_max_value=self.pointer_gauge_max_spin.value() if object_type == "gauge" else None,
        )
        if not saved:
            self.set_status_message(result)
            return
        self.load_pointer_records()
        type_name = self.current_pointer_object_name()
        rules_path = os.path.join(self.output_dir, "pointer_panel_rules.json")
        self.set_status_message(
            f"状态：已保存{type_name}对象 {result.get('panel_type') if object_type == 'gauge' else (result.get('label') or result.get('text') or '')} "
            f"{result.get('angle_deg'):.1f}°，并更新面板规则：{rules_path}"
        )

    def delete_pointer_record_by_index(self, index):
        if index < 0 or index >= len(self.pointer_records):
            return
        record = self.pointer_records[index]
        if delete_pointer_record(self.output_dir, record, self.input_dir):
            self.load_pointer_records()
            self.set_status_message("状态：已删除角度识别标注。")
        else:
            self.set_status_message("状态：未找到要删除的角度识别标注。")

    def delete_selected_pointer_record(self):
        record = self.selected_pointer_record()
        if record is None:
            self.set_status_message("状态：请先选择一条角度识别标注。")
            return
        if delete_pointer_record(self.output_dir, record, self.input_dir):
            self.load_pointer_records()
            self.set_status_message("状态：已删除角度识别标注。")
        else:
            self.set_status_message("状态：未找到要删除的角度识别标注。")

    def update_selected_pointer_record(self):
        record = self.selected_pointer_record()
        if record is None:
            self.set_status_message("状态：请先选择一条角度识别标注。")
            return

        valid, message = self.validate_selected_pointer_type()
        if not valid:
            self.set_status_message(message)
            return
        object_type = self.current_pointer_rule_type()
        saved, result = update_pointer_record_metadata(
            self.output_dir,
            self.input_dir,
            record,
            label=self.selected_pointer_label(),
            text=self.pointer_text_edit.text().strip(),
            panel_type=self.selected_pointer_panel_type(),
            object_name=self.current_pointer_object_name(),
            object_type=object_type,
            gauge_value=self.pointer_gauge_value_spin.value() if object_type == "gauge" else None,
            gauge_min_value=self.pointer_gauge_min_spin.value() if object_type == "gauge" else None,
            gauge_max_value=self.pointer_gauge_max_spin.value() if object_type == "gauge" else None,
        )
        if not saved:
            self.set_status_message(result)
            return

        self.load_pointer_records()
        rules_path = os.path.join(self.output_dir, "pointer_panel_rules.json")
        self.set_status_message(f"状态：已按 panel_type 更新历史标注，并重新生成规则：{rules_path}")

    def selected_class_parts(self):
        if not self.current_class_path:
            return []
        return [part for part in self.current_class_path.split("/") if part]

    def selected_class_leaf(self):
        parts = self.selected_class_parts()
        return parts[-1] if parts else ""

    def selected_class_flat_name(self):
        return flat_class_name(self.current_class_path or "")

    def selected_pointer_parts(self):
        if not self.current_pointer_path:
            return []
        return [part for part in self.current_pointer_path.split("/") if part]

    def selected_pointer_gauge_parts(self):
        if not self.current_pointer_gauge_path:
            return []
        return [part for part in self.current_pointer_gauge_path.split("/") if part]

    def selected_pointer_panel_type(self):
        if self.current_pointer_rule_type() == "gauge":
            return "/".join(self.selected_pointer_gauge_parts())
        parts = self.selected_pointer_parts()
        return "/".join(parts[:-1]) if len(parts) >= 2 else ""

    def selected_pointer_label(self):
        if self.current_pointer_rule_type() == "gauge":
            return ""
        parts = self.selected_pointer_parts()
        return parts[-1] if len(parts) >= 2 else ""

    def validate_selected_pointer_type(self):
        if not self.current_pointer_object_name():
            return False, "状态：请先新增或填写一个角度对象种类。"
        if self.current_pointer_rule_type() == "gauge":
            if not self.selected_pointer_gauge_parts():
                return False, "状态：请在指针读数表类型树中选择一个表类型。"
            selected_item = self.pointer_gauge_tree_panel.selected_item()
            if selected_item is not None and selected_item.childCount() > 0:
                return False, "状态：请选择指针读数表类型树的叶子节点。"
            return True, ""

        parts = self.selected_pointer_parts()
        if len(parts) < 2:
            return False, "状态：请在旋钮档位树中选择一个叶子档位，例如 旋钮类型/档位语义。"
        selected_item = self.pointer_tree_panel.selected_item()
        if selected_item is not None and selected_item.childCount() > 0:
            return False, "状态：请选择旋钮档位树的叶子节点。"
        return True, ""

    def get_crop_bounds(self):
        rect = self.image_label.get_selection_rect()
        return self.bounds_from_rect(rect)

    def bounds_from_rect(self, rect):
        if rect.isNull() or self.img is None:
            return None

        img_height, img_width = self.img.shape[:2]
        x1 = clamp(rect.left(), 0, img_width - 1)
        y1 = clamp(rect.top(), 0, img_height - 1)
        x2 = clamp(rect.left() + rect.width(), 0, img_width)
        y2 = clamp(rect.top() + rect.height(), 0, img_height)
        if x2 <= x1 or y2 <= y1:
            return None

        return x1, y1, x2, y2

    def update_crop_preview(self):
        if self.mode != MODE_CROP:
            return

        if self.classification_full_image:
            self.crop_label.setPixmap(QPixmap())
            self.crop_label.setText("整图分类：保存当前完整图片")
            self.selection_info.setText("分类样本：整图，不裁剪")
            return

        if self.crop_boxes:
            self.crop_label.setPixmap(QPixmap())
            self.crop_label.setText(f"待保存裁剪框：{len(self.crop_boxes)}")
            self.selection_info.setText(f"裁剪框：{len(self.crop_boxes)} 个待保存")
            return

        bounds = self.get_crop_bounds()
        if bounds is None:
            self.crop_label.setPixmap(QPixmap())
            self.crop_label.setText("拖画左侧原图后，这里显示裁剪预览")
            self.selection_info.setText("裁剪框：未选择")
            return

        x1, y1, x2, y2 = bounds
        crop = self.img[y1:y2, x1:x2]
        if crop.size == 0:
            self.crop_label.setPixmap(QPixmap())
            self.crop_label.setText("裁剪区域无效")
            self.selection_info.setText("裁剪框：无效")
            return

        preview_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        height, width, channel = preview_rgb.shape
        qimg = QImage(
            preview_rgb.data, width, height, channel * width, QImage.Format_RGB888
        ).copy()
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.crop_label.width() - 12,
            self.crop_label.height() - 12,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.crop_label.setPixmap(pixmap)
        self.crop_label.setText("")
        self.selection_info.setText(
            f"裁剪框：左上 ({x1}, {y1})，右下 ({x2}, {y2})，尺寸 {x2 - x1} x {y2 - y1}"
        )

    def save_current_crop(self):
        if self.img is None:
            return False, "状态：当前没有可保存的图片。"

        if self.classification_full_image:
            class_valid, message = self.validate_selected_class()
            if not class_valid:
                return False, message
            img_height, img_width = self.img.shape[:2]
            saved, message = self.save_classification_region((0, 0, img_width, img_height))
            if saved:
                self.load_crop_history_records()
            return saved, message

        if not self.crop_boxes:
            return False, "状态：请先在左侧原图上拖画至少一个裁剪框。"

        saved_count = 0
        for box in list(self.crop_boxes):
            saved, message = self.save_classification_region(
                (box["x1"], box["y1"], box["x2"], box["y2"]),
                box["class_path"],
            )
            if not saved:
                return False, message
            saved_count += 1

        self.crop_boxes = []
        self.image_label.set_crop_boxes(self.crop_boxes)
        self.load_crop_history_records()
        self.update_crop_preview()
        return True, f"状态：已保存 {saved_count} 个裁剪框。"

    def crop_history_box_from_record(self, record):
        bounds = record.get("bbox_xyxy") or []
        if len(bounds) != 4:
            return None
        try:
            x1, y1, x2, y2 = [int(value) for value in bounds]
        except (TypeError, ValueError):
            return None
        class_path = record.get("class_path") or ""
        return {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "class_path": class_path,
            "class_leaf": class_path.split("/")[-1] if class_path else "",
        }

    def crop_history_matches_current_image(self, record):
        return record.get("image_rel") == self.current_image_rel

    def load_crop_history_records(self):
        if not hasattr(self, "crop_history_list"):
            return
        self.crop_history_list.blockSignals(True)
        self.crop_history_list.clear()
        all_records = load_manifest(self.output_dir)
        self.crop_history_records = [
            record for record in all_records if self.crop_history_matches_current_image(record)
        ]
        paired_count = self.count_paired_history_records(all_records)
        history_boxes = []
        for index, record in enumerate(self.crop_history_records):
            box = self.crop_history_box_from_record(record)
            if box:
                history_boxes.append(box)
            bounds = record.get("bbox_xyxy") or []
            class_path = record.get("class_path") or "未分类"
            self.crop_history_list.addItem(f"{index + 1}. {class_path} {bounds}")
        if not self.crop_history_records:
            self.crop_history_list.addItem("当前图片没有历史裁剪标注")
            if paired_count:
                self.crop_history_list.addItem(f"配对原图/result 图有 {paired_count} 条历史，未自动混入")
        self.crop_history_list.blockSignals(False)
        self.image_label.set_crop_history_boxes(history_boxes)
        self.update_crop_preview()

    def history_pair_key(self, image_rel):
        normalized = os.path.normcase(os.path.normpath(image_rel or ""))
        root, ext = os.path.splitext(normalized)
        if root.endswith("_result"):
            root = root[:-7]
        return root + ext

    def count_paired_history_records(self, records):
        current_key = self.history_pair_key(self.current_image_rel)
        return sum(
            1
            for record in records
            if record.get("image_rel") != self.current_image_rel
            and self.history_pair_key(record.get("image_rel")) == current_key
        )

    def selected_crop_history_record(self):
        if not hasattr(self, "crop_history_list"):
            return None
        row = self.crop_history_list.currentRow()
        if row < 0 or row >= len(self.crop_history_records):
            return None
        return self.crop_history_records[row]

    def select_crop_history_record(self, row):
        if row < 0 or row >= len(self.crop_history_records):
            return
        record = self.crop_history_records[row]
        bounds = record.get("bbox_xyxy") or []
        self.selection_info.setText(
            f"历史裁剪框：{record.get('class_path', '')} {bounds}"
        )

    def reclassify_selected_crop_history(self):
        record = self.selected_crop_history_record()
        if record is None:
            self.set_status_message("状态：请先选择一条历史裁剪标注。")
            return
        class_valid, message = self.validate_selected_class()
        if not class_valid:
            self.set_status_message(message)
            return
        if self.img is None:
            self.set_status_message("状态：当前没有可修改的图片。")
            return
        if record.get("class_path") == self.current_class_path:
            self.set_status_message("状态：该历史标注已经是当前分类，无需修改。")
            return

        saved, result = update_manifest_record_class(
            self.output_dir,
            self.input_dir,
            record,
            self.img,
            self.current_class_path,
        )
        if not saved:
            self.set_status_message(result)
            return
        self.load_crop_history_records()
        self.set_status_message(f"状态：已把历史标注改为 {result.get('class_path')}。")

    def replace_selected_crop_history(self):
        record = self.selected_crop_history_record()
        if record is None:
            self.set_status_message("状态：请先选择一条历史裁剪标注。")
            return
        class_valid, message = self.validate_selected_class()
        if not class_valid:
            self.set_status_message(message)
            return
        if self.img is None:
            self.set_status_message("状态：当前没有可修改的图片。")
            return

        bounds = self.get_crop_bounds()
        if bounds is None:
            self.set_status_message("状态：请先在原图上拖出新的裁剪框。")
            return

        saved, result = update_manifest_record(
            self.output_dir,
            self.input_dir,
            record,
            self.img,
            bounds,
            self.current_class_path,
        )
        if not saved:
            self.set_status_message(result)
            return
        self.image_label.clear_selection()
        self.load_crop_history_records()
        self.set_status_message(f"状态：已用新框替换历史标注，分类为 {result.get('class_path')}。")

    def delete_selected_crop_history(self):
        record = self.selected_crop_history_record()
        if record is None:
            self.set_status_message("状态：请先选择一条历史裁剪标注。")
            return

        if delete_manifest_record(self.output_dir, self.input_dir, record):
            self.load_crop_history_records()
            self.set_status_message("状态：已删除历史裁剪标注，并更新导出数据。")
        else:
            self.set_status_message("状态：未找到要删除的历史裁剪标注。")

    def validate_selected_class(self):
        if not self.selected_class_parts():
            return False, "状态：请先在分类树中选择一个分类，再保存。"

        selected_item = self.class_tree_panel.selected_item()
        if selected_item is not None and selected_item.childCount() > 0:
            return False, "状态：请选择分类树的叶子节点，再保存。"
        return True, ""

    def save_classification_region(self, bounds, class_path=None):
        return save_crop(
            self.output_dir,
            self.input_dir,
            self.current_image_rel,
            self.img,
            bounds,
            class_path or self.current_class_path,
        )

    def add_crop_box(self, rect):
        if self.img is None:
            return
        if self.classification_full_image:
            self.set_status_message("状态：整图分类模式下不需要拖框，直接点击保存分类样本。")
            return

        class_valid, message = self.validate_selected_class()
        if not class_valid:
            self.set_status_message(message)
            return

        bounds = self.bounds_from_rect(rect)
        if bounds is None:
            return

        x1, y1, x2, y2 = bounds
        self.crop_boxes.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "class_path": self.current_class_path,
                "class_leaf": self.selected_class_leaf(),
            }
        )
        self.image_label.set_crop_boxes(self.crop_boxes)
        self.update_crop_preview()
        self.set_status_message(f"状态：已添加裁剪框 {len(self.crop_boxes)} 个。")

    def delete_crop_box(self, index):
        if index < 0 or index >= len(self.crop_boxes):
            return
        removed = self.crop_boxes.pop(index)
        self.image_label.set_crop_boxes(self.crop_boxes)
        self.update_crop_preview()
        self.set_status_message(f"状态：已删除裁剪框 {removed.get('class_leaf', '')}。")

    def business_bounds_from_rect(self, rect):
        return self.bounds_from_rect(rect)

    def load_business_records(self):
        if not hasattr(self, "business_history_list"):
            return
        self.business_history_list.blockSignals(True)
        self.business_history_list.clear()
        self.business_records = [
            record for record in load_business_manifest(self.output_dir)
            if record.get("image_rel") == self.current_image_rel
        ]
        boxes = []
        for index, record in enumerate(self.business_records):
            bbox = record.get("bbox_xyxy") or []
            component_type = record.get("component_type") or ""
            biz_name = record.get("biz_name") or ""
            if len(bbox) == 4:
                boxes.append(
                    {
                        "x1": int(bbox[0]),
                        "y1": int(bbox[1]),
                        "x2": int(bbox[2]),
                        "y2": int(bbox[3]),
                        "component_type": component_type,
                        "biz_name": biz_name,
                    }
                )
            self.business_history_list.addItem(
                f"{index + 1}. {component_type} | {biz_name} bbox:{bbox}"
            )
        if not self.business_records:
            self.business_history_list.addItem("当前图片没有业务标注")
        self.business_history_list.blockSignals(False)
        self.image_label.set_business_boxes(boxes)

    def selected_business_record(self):
        if not hasattr(self, "business_history_list"):
            return None
        row = self.business_history_list.currentRow()
        if row < 0 or row >= len(self.business_records):
            return None
        return self.business_records[row]

    def select_business_record(self, row):
        if row < 0 or row >= len(self.business_records):
            return
        record = self.business_records[row]
        component_type = record.get("component_type") or ""
        biz_name = record.get("biz_name") or ""
        self.business_tree_panel.select_path(f"{component_type}/{biz_name}" if biz_name else component_type)
        self.selection_info.setText(
            f"业务标注：{component_type} | {biz_name} {record.get('bbox_xyxy') or []}"
        )

    def add_business_box(self, rect):
        if self.img is None or not self.current_image_rel:
            return
        bounds = self.business_bounds_from_rect(rect)
        if bounds is None:
            return
        saved, result = add_business_record(
            self.output_dir,
            self.input_dir,
            self.current_image_rel,
            self.img,
            bounds,
            self.current_business_component_type,
            self.current_business_biz_name,
            self.current_business_note,
        )
        if not saved:
            self.set_status_message(result)
            return
        self.load_business_records()
        self.set_status_message(
            f"状态：已保存业务标注 {result.get('component_type')} | {result.get('biz_name')}。"
        )

    def delete_business_record_by_index(self, index):
        if index < 0 or index >= len(self.business_records):
            return
        record = self.business_records[index]
        if delete_business_record(self.output_dir, self.input_dir, record):
            self.load_business_records()
            self.set_status_message("状态：已删除业务标注。")
        else:
            self.set_status_message("状态：未找到要删除的业务标注。")

    def delete_selected_business_record(self):
        record = self.selected_business_record()
        if record is None:
            self.set_status_message("状态：请先选择一条业务标注。")
            return
        if delete_business_record(self.output_dir, self.input_dir, record):
            self.load_business_records()
            self.set_status_message("状态：已删除业务标注。")
        else:
            self.set_status_message("状态：未找到要删除的业务标注。")

    def update_selected_business_record(self):
        record = self.selected_business_record()
        if record is None:
            self.set_status_message("状态：请先选择一条业务标注。")
            return
        saved, result = update_business_record(
            self.output_dir,
            self.input_dir,
            record,
            self.current_business_component_type,
            self.current_business_biz_name,
            self.current_business_note,
        )
        if not saved:
            self.set_status_message(result)
            return
        self.load_business_records()
        self.set_status_message(
            f"状态：已更新业务标注 {result.get('component_type')} | {result.get('biz_name')}。"
        )

    def save(self):
        if self.mode == MODE_DETECT:
            saved, message = self.save_current_labels()
            self.set_status_message(message)
            return

        if self.mode == MODE_POINTER:
            self.set_status_message("状态：旋钮对象标注会在点完中心和尖端后自动保存。")
            return

        if self.mode == MODE_BUSINESS:
            self.set_status_message("状态：业务标注会在拖完组件框后自动保存。")
            return

        saved, message = self.save_current_crop()
        if saved:
            self.image_label.clear_selection()
        self.set_status_message(message)
