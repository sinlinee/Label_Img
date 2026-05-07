import json
import os

from .constants import BUSINESS_COMPONENT_TYPES_PATH, BUSINESS_TREE_PATH


def normalize_business_tree(nodes):
    normalized = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        name = str(node.get("name", "")).strip()
        if not name:
            continue
        normalized.append(
            {
                "name": name,
                "note": str(node.get("note", "")).strip(),
                "children": normalize_business_tree(node.get("children", [])),
            }
        )
    return normalized


def load_business_tree_config():
    if not os.path.exists(BUSINESS_TREE_PATH):
        return []
    try:
        with open(BUSINESS_TREE_PATH, "r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
    except (OSError, json.JSONDecodeError):
        return []
    return normalize_business_tree(data) if isinstance(data, list) else []


def save_business_tree_config(tree_data):
    cleaned = normalize_business_tree(tree_data)
    with open(BUSINESS_TREE_PATH, "w", encoding="utf-8") as file_obj:
        json.dump(cleaned, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
    return cleaned


def load_business_component_types():
    tree_types = [node["name"] for node in load_business_tree_config()]
    legacy_types = []
    if os.path.exists(BUSINESS_COMPONENT_TYPES_PATH):
        try:
            with open(BUSINESS_COMPONENT_TYPES_PATH, "r", encoding="utf-8") as file_obj:
                data = json.load(file_obj)
        except (OSError, json.JSONDecodeError):
            data = []
        if isinstance(data, list):
            legacy_types = [str(item).strip() for item in data if str(item).strip()]
    merged = []
    for name in tree_types + legacy_types:
        if name and name not in merged:
            merged.append(name)
    return merged


def save_business_component_types(component_types):
    cleaned = []
    seen = set()
    for item in component_types:
        name = str(item).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        cleaned.append(name)
    with open(BUSINESS_COMPONENT_TYPES_PATH, "w", encoding="utf-8") as file_obj:
        json.dump(cleaned, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
    return cleaned
