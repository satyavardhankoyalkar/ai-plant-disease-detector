import numpy as np
import json
from keras.models import load_model
from pathlib import Path

model = None
labels = None


def load_once():
    global model, labels

    if model is None:
        print("🌿 Loading model (.keras format)...")

        # Resolve absolute paths (Render safe)
        base = Path(__file__).resolve().parent.parent
        model_path = base / "plant_model_fixed.keras"
        labels_path = base / "labels.json"

        # Load model
        model = load_model(model_path, compile=False)

        # Load labels
        with open(labels_path) as f:
            labels = json.load(f)


def predict(arr):
    load_once()

    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))

    return labels[idx], float(preds[idx] * 100)