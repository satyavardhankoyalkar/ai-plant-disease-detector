
import json
import tensorflow as tf
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "plant_model.h5")
LABEL_PATH = os.path.join(BASE_DIR, "labels.json")

# Lazy load
model = None
labels = None

def load():
    global model, labels
    if model is None:
        print("🔥 Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        labels = json.load(open(LABEL_PATH))
        print("✅ Model loaded")

def predict(img_array):
    load()
    preds = model.predict(img_array)[0]
    idx = preds.argmax()
    return labels[idx], float(preds[idx] * 100)