import json
import os

from .constants import SETTINGS_PATH


def normalize_user_path(path):
    path = path.strip().strip('"').strip("'")
    if not path:
        return ""
    path = os.path.expandvars(os.path.expanduser(path))
    return os.path.abspath(path)


def load_settings():
    if not os.path.exists(SETTINGS_PATH):
        return {}

    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as file_obj:
            settings = json.load(file_obj)
    except (OSError, json.JSONDecodeError):
        return {}

    return settings if isinstance(settings, dict) else {}


def save_settings(settings):
    with open(SETTINGS_PATH, "w", encoding="utf-8") as file_obj:
        json.dump(settings, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
