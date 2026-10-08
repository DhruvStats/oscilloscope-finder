"""Load config/instruments.yaml - the single list of recognised instruments (see the comments in that file)."""
import os
import re

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "config", "instruments.yaml")
FACES = ("front", "back", "left", "right", "top", "bottom")


def load(path=PATH):
    with open(path, encoding="utf-8") as f:
        items = yaml.safe_load(f)["instruments"]
    seen = set()
    for it in items:
        label = it["label"]
        if not re.fullmatch(r"[a-z0-9_]+", label) or label in seen:
            raise ValueError(f"instrument label {label!r} must be unique, lowercase letters/digits/underscores")
        seen.add(label)
        it.setdefault("display_name", label)
        it.setdefault("colour", "#18c27f")
        it.setdefault("width_mm", 350)
        it.setdefault("sample_weight", 1.0)
        it.setdefault("hints", "")
        it.setdefault("textures", [])
        for variant in it["textures"]:
            missing = set(FACES) - set(variant)
            if missing:
                raise ValueError(f"{label}: texture variant is missing faces {sorted(missing)}")
    return items


def labels(path=PATH):
    return [it["label"] for it in load(path)]
