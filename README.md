# 🌿 AI Plant Disease Detection

An AI-powered web application that detects plant diseases from leaf images using deep learning.  
Built with **Flask + TensorFlow**, this project demonstrates end-to-end machine learning deployment.

---
#Model Download link:https://drive.google.com/file/d/1ioGCHoPyl3sdw16n7ODtWoQuq94wPtWw/view?usp=sharing

---
<img width="1016" height="574" alt="image" src="https://github.com/user-attachments/assets/3cddcd1c-3393-4b2c-9e25-121c4d805cc3" />
<img width="889" height="842" alt="image" src="https://github.com/user-attachments/assets/04d22f3b-6691-4ab7-b01d-4f65113b992d" />


## 🚀 Features

- 🌱 Upload leaf image and detect disease
- 🤖 Deep learning CNN-based prediction
- 📊 Confidence score display
- 🖥️ Clean and simple web interface
- ⚡ Real-time inference using Flask

---

## 🧠 Tech Stack

- **Backend:** Flask (Python)
- **ML Model:** TensorFlow / Keras CNN
- **Frontend:** HTML + CSS
- **Image Processing:** Pillow / NumPy

---

## 📸 Demo

> Upload a plant leaf image to analyze plant health.

Example Output:

- Disease: Tomato Target Spot  
- Confidence: 92.4%

⚠️ Note: This is a prototype model and results may vary.

---

## 🏗️ Project Structure

```
plant-ai/
│
├── app.py                # Flask backend
├── plant_model.h5        # Trained CNN model
├── requirements.txt
│
├── templates/
│   └── index.html        # Frontend UI
│
└── static/
    └── uploads/          # Uploaded images
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-plant-disease-detector.git
cd ai-plant-disease-detector
```

### 2️⃣ Create virtual environment

```bash
python -m venv tfenv
tfenv\Scripts\activate   # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the app

```bash
python app.py
```

Open in browser:

```
http://127.0.0.1:5000
```

---

## 🧪 How It Works

1. User uploads a leaf image
2. Image is resized and normalized
3. CNN model predicts disease class
4. Result + confidence displayed on UI

---

## 📌 Future Improvements

- 🌿 Better dataset training
- 📱 Mobile responsive UI
- 🔥 Grad-CAM heatmap visualization
- ☁️ Cloud deployment (Render / Railway)
- 🌍 Multi-language support

---

## ⚠️ Disclaimer

This project is built for **educational and demonstration purposes**.  
The model is trained on a limited dataset and may not generalize to all real-world plant species.

---

## 👨‍💻 Author

**Satya**  
Aspiring Software Developer | AI & Web Enthusiast

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and share feedback!
