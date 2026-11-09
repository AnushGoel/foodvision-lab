"""Design tokens shared by the interactive charts (``viz.py``) and the static summary figure (``figures.py``).

Model size is ordinal, so B2 -> B4 use a light-to-dark blue ramp. Classes use food colours (tomato, char, nori).
Saffron marks the selected model and chosen operating points. Corruptions get muted hues outside both palettes.
"""

INK = "#1B2430"
MUTED = "#5B6573"
GRID = "#E6E9EE"
SAFFRON = "#E3A21A"
NON_FOOD = "#B3BAC4"
MODEL_COLORS = {"effnetb2": "#9DB7DA", "effnetb3": "#4F75B3", "effnetb4": "#1F3A68"}
MODEL_LABELS = {"effnetb2": "EfficientNet-B2", "effnetb3": "EfficientNet-B3", "effnetb4": "EfficientNet-B4"}
CLASS_COLORS = {"pizza": "#C8402F", "steak": "#7A4E36", "sushi": "#2F6B5B"}
METRIC_COLORS = {"precision": "#9DB7DA", "recall": "#4F75B3", "f1": "#1F3A68"}
OOD_COLORS = {"food": INK, "non-food": NON_FOOD}
CORRUPTION_COLORS = {"blur": "#7E6BC4", "noise": "#B5566B", "dark": "#4A4F57", "jpeg": "#C47A2C", "lowres": "#3E8E9E"}
