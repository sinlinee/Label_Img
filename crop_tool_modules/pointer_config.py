import json
import os

from .common_utils import clone_class_tree
from .constants import POINTER_GAUGE_TREE_PATH, POINTER_OBJECT_TYPES_PATH, POINTER_TREE_PATH


def load_pointer_tree_config():
    if not os.path.exists(POINTER_TREE_PATH):
        return []
    try:
        with open(POINTER_TREE_PATH, "r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
    except (OSError, json.JSONDecodeError):
        return []
    return clone_class_tree(data) if isinstance(data, list) else []


def load_pointer_gauge_tree_config():
    if not os.path.exists(POINTER_GAUGE_TREE_PATH):
        return []
    try:
        with open(POINTER_GAUGE_TREE_PATH, "r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
    except (OSError, json.JSONDecodeError):
        return []
    return clone_class_tree(data) if isinstance(data, list) else []


def load_pointer_object_types():
    if not os.path.exists(POINTER_OBJECT_TYPES_PATH):
        return []
    try:
        with open(POINTER_OBJECT_TYPES_PATH, "r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    object_types = []
    for item in data:
        name = str(item.get("name", "") if isinstance(item, dict) else item).strip()
        rule_type = str(item.get("rule_type", "") if isinstance(item, dict) else "").strip()
        if name:
            object_types.append({"name": name, "rule_type": rule_type or name})
    return object_types


def save_pointer_object_types(object_types):
    cleaned = []
    seen = set()
    for item in object_types:
        name = str(item.get("name", "") if isinstance(item, dict) else item).strip()
        rule_type = str(item.get("rule_type", "") if isinstance(item, dict) else "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        cleaned.append({"name": name, "rule_type": rule_type or name})
    with open(POINTER_OBJECT_TYPES_PATH, "w", encoding="utf-8") as file_obj:
        json.dump(cleaned, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
    return cleaned
