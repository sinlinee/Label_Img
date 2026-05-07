import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT_DIR = os.path.join(BASE_DIR, "images")
REFER_DIR = os.path.join(BASE_DIR, "refer")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "dataset")
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "yolo11n.pt")
SETTINGS_PATH = os.path.join(BASE_DIR, "crop_tool_settings.json")
CLASS_TREE_PATH = os.path.join(BASE_DIR, "classification_tree.json")
POINTER_TREE_PATH = os.path.join(BASE_DIR, "pointer_tree.json")
POINTER_GAUGE_TREE_PATH = os.path.join(BASE_DIR, "pointer_gauge_tree.json")
POINTER_OBJECT_TYPES_PATH = os.path.join(BASE_DIR, "pointer_object_types.json")
BUSINESS_COMPONENT_TYPES_PATH = os.path.join(BASE_DIR, "business_component_types.json")
BUSINESS_TREE_PATH = os.path.join(BASE_DIR, "business_tree.json")

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
MODE_CROP = "crop"
MODE_DETECT = "detect"
MODE_POINTER = "pointer"
MODE_BUSINESS = "business"
CLASS_MANIFEST_NAME = "classification_annotations.jsonl"
POINTER_MANIFEST_NAME = "pointer_annotations.jsonl"
BUSINESS_MANIFEST_NAME = "business_annotations.jsonl"

DEFAULT_CLASS_TREE = [
    {
        "name": "旋钮",
        "children": [
            {"name": "类型1", "children": [{"name": "开"}, {"name": "关"}]},
            {"name": "类型2", "children": [{"name": "远方"}, {"name": "0"}, {"name": "接地"}]},
        ],
    },
    {
        "name": "Voltage显示",
        "children": [{"name": "类型1", "children": [{"name": "接地亮"}, {"name": "接地不亮"}]}],
    },
]
