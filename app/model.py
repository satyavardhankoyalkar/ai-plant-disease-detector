import numpy as np
import json
import tensorflow as tf

model = None
labels = None

def load_once():
    global model, labels
    if model is None:
        print("🌿 Loading TF model...")
        model = tf.keras.models.load_model("plant_model.h5", compile=False)

        with open("labels.json") as f:
            labels = json.load(f)

def predict(arr):
    load_once()
    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    return labels[idx], float(preds[idx] * 100)
