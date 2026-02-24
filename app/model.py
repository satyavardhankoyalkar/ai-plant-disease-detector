import os
import numpy as np
from tensorflow.keras.models import load_model

model = None  # lazy global

def get_model():
    global model
    if model is None:
        BASE_DIR = os.path.dirname(os.path.dirname(__file__))
        MODEL_PATH = os.path.join(BASE_DIR, "plant_model.h5")
        model = load_model(MODEL_PATH)
    return model


def predict(image):
    model = get_model()

    image = image.resize((224, 224))
    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    idx = np.argmax(preds)

    classes = [
        "Tomato Early Blight",
        "Tomato Late Blight",
        "Tomato Healthy",
        # add your labels here
    ]

    return classes[idx], float(np.max(preds))