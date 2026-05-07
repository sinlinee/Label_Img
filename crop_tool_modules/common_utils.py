def clamp(value, low, high):
    return max(low, min(value, high))


def copy_boxes(boxes):
    return [dict(box) for box in boxes]


def safe_path_part(text):
    cleaned = "".join("_" if char in '<>:"/\\|?*' else char for char in str(text).strip())
    cleaned = cleaned.strip(" .")
    return cleaned or "unnamed"


def clone_class_tree(nodes):
    return [
        {
            "name": str(node.get("name", "未命名")),
            "children": clone_class_tree(node.get("children", [])),
        }
        for node in nodes
        if isinstance(node, dict)
    ]
