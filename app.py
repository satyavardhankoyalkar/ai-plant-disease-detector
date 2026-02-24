import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

st.set_page_config(
    page_title="🌿 Plant Disease Detector",
    page_icon="🌿",
    layout="centered"
)

CLASS_NAMES = {
    0:  "Pepper Bell - Bacterial Spot",
    1:  "Pepper Bell - Healthy",
    2:  "Potato - Early Blight",
    3:  "Potato - Late Blight",
    4:  "Potato - Healthy",
    5:  "Tomato - Bacterial Spot",
    6:  "Tomato - Early Blight",
    7:  "Tomato - Late Blight",
    8:  "Tomato - Leaf Mold",
    9:  "Tomato - Septoria Leaf Spot",
    10: "Tomato - Spider Mites (Two Spotted)",
    11: "Tomato - Target Spot",
    12: "Tomato - Yellow Leaf Curl Virus",
    13: "Tomato - Mosaic Virus",
    14: "Tomato - Healthy",
}

@st.cache_resource
def load_my_model():
    return load_model("plant_model.h5")

model = load_my_model()

def predict(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    pred = model.predict(img_array)[0]
    idx = int(np.argmax(pred))
    confidence = float(pred[idx] * 100)
    return idx, confidence, pred

# ── UI ───────────────────────────────────────────────────────────────────────
st.title("🌿 AI Plant Disease Detector")
st.markdown("Upload a **leaf image** of Tomato, Potato, or Pepper to detect disease.")
st.divider()

uploaded_file = st.file_uploader("📷 Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded Image", use_column_width=True)

    with col2:
        with st.spinner("🔍 Analyzing..."):
            idx, confidence, probs = predict(img)

        disease = CLASS_NAMES[idx]
        is_healthy = "healthy" in disease.lower()
        is_unknown = confidence < 70

        st.markdown("### 🔬 Result")

        if is_unknown:
            st.warning("⚠️ Unknown / Unsupported Plant")
            st.info("Please upload a Tomato, Potato, or Pepper leaf image.")
        elif is_healthy:
            st.success(f"✅ **{disease}**")
            st.balloons()
        else:
            st.error(f"⚠️ **{disease}**")

        st.markdown("### 📊 Confidence")
        st.progress(int(confidence))
        st.metric("Confidence Score", f"{confidence:.2f}%")

    st.divider()

    with st.expander("📈 See all class probabilities"):
        for i, p in enumerate(probs):
            st.write(f"**{CLASS_NAMES[i]}** — {p*100:.2f}%")
            st.progress(float(p))

else:
    st.info("👆 Upload a leaf image to get started!")
    st.markdown("### 🌱 Supported Plants & Diseases")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🫑 Pepper Bell**")
        st.markdown("- Bacterial Spot\n- Healthy")

    with col2:
        st.markdown("**🥔 Potato**")
        st.markdown("- Early Blight\n- Late Blight\n- Healthy")

    with col3:
        st.markdown("**🍅 Tomato**")
        st.markdown("- Bacterial Spot\n- Early Blight\n- Late Blight\n- Leaf Mold\n- Septoria Leaf Spot\n- Spider Mites\n- Target Spot\n- Yellow Leaf Curl\n- Mosaic Virus\n- Healthy")

st.divider()
st.markdown(
    "<center>Made by <b>Satya</b> 🌿 | MobileNetV2 + TensorFlow</center>",
    unsafe_allow_html=True
)
