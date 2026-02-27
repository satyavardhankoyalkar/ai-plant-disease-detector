# 🌿 AI Plant Disease Detector (V3)

An AI-powered web app that detects plant diseases from leaf images using deep learning.
Built with TensorFlow and deployed on Hugging Face Spaces using Gradio.

---

## 🚀 Live Demo

👉 https://huggingface.co/spaces/mobuto/plant-disease-detector

Upload a plant leaf image and instantly get:

* 🌱 Disease prediction
* 📊 Confidence score

---

## ✨ Features

* 🧠 Deep Learning based image classification
* 🌿 Supports 14+ plant disease classes
* ⚡ Real-time predictions
* 🌐 Fully deployed web app (no install needed)
* 📦 Lightweight Gradio interface

---

## 🛠 Tech Stack

* **Model:** TensorFlow / Keras (MobileNet-based CNN)
* **Frontend:** Gradio
* **Deployment:** Hugging Face Spaces
* **Language:** Python

---

## 📂 Project Structure

```
ai-plant-disease-detector/
│
├── app.py              # Gradio web app
├── saved_model/        # TensorFlow SavedModel
├── labels.json         # Class labels
├── requirements.txt    # Dependencies
├── runtime.txt         # Python version
└── README.md
```

---

## 🧠 Model Details

* Input size: **224x224 RGB**
* Architecture: Transfer Learning (CNN backbone)
* Dataset: PlantVillage dataset
* Classes: 14+ plant diseases + healthy categories

---

## 📸 Example Diseases Detected

* Tomato Early Blight
* Potato Late Blight
* And more...

---

## ⚙️ Run Locally

```bash
git clone https://github.com/satyavardhankoyalkar/ai-plant-disease-detector
cd ai-plant-disease-detector

pip install -r requirements.txt
python app.py
```

---

## 🎯 Use Cases

* Farmers & agriculture students
* Plant disease awareness tools
* AI in agriculture demos
* Computer vision projects

---

## 📈 Future Improvements

* Mobile app version 📱
* Disease treatment suggestions 💊
* Multi-language support 🌍
* API deployment 🔌

---

## 👨‍💻 Author

**Satyavardhan Koyalkar**
B.Tech CSE Student | AI & Full Stack Enthusiast

GitHub:
https://github.com/satyavardhankoyalkar

---

## ⭐ Support

If you found this project helpful:

* ⭐ Star the repo
* 🍴 Fork it
* 📢 Share with others

---

## 📜 License

This project is for educational and research purposes.
