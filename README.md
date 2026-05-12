# 🐶 Dog Breed Classifier

![Streamlit](https://img.shields.io/badge/Streamlit-1.42.0-red?style=flat-square&logo=streamlit)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-orange?style=flat-square&logo=pytorch)
![Python](https://img.shields.io/badge/Python-3.11+-3776ab?style=flat-square&logo=python)
![EfficientNetB0](https://img.shields.io/badge/Model-EfficientNetB0-blue?style=flat-square)
![Breeds](https://img.shields.io/badge/Breeds-55-green?style=flat-square)

---

## 🚀 Overview

A lightweight Streamlit web app for dog breed classification using **PyTorch** and **EfficientNetB0**. The app performs image inference with a trained model stored in `models/best_efficientnet_b0.pth` and identifies **55 dog breeds** from a single upload or camera input.

---

## ✨ Key features

- **55 dog breeds** recognized
- **EfficientNetB0** with PyTorch inference
- **Streamlit UI** with bilingual support (English / Español)
- **Top 5 predictions** with confidence scores
- **Camera upload** and image upload support
- **GPU / CPU-aware** inference
- **Production-ready** repo structure for GitHub and Streamlit Cloud

---

## 🧠 Model

- Architecture: **EfficientNetB0**
- Framework: **PyTorch**
- Model file: `models/best_efficientnet_b0.pth`
- Classes: **55 dog breeds**
- Input size: **224×224 RGB**

---

## ⚙️ Technology stack

- `streamlit`
- `torch`
- `torchvision`
- `numpy`
- `pandas`
- `plotly`
- `Pillow`

---

## 🚀 Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📂 Project structure

```text
dog-breed-classifier/
├── app.py
├── models/
│   └── best_efficientnet_b0.pth
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📌 Notes

- The app uses the trained PyTorch model from `models/best_efficientnet_b0.pth`.
- It is designed for **easy deployment on Streamlit Cloud**.
- Predictions are shown with a **Top 5 chart** and confidence metrics.

---

## ✅ Current status

This repository is now focused on the production-ready application: clean, minimal, and ready for deployment. Legacy training and dataset files have been moved out of the root structure to keep the repository compact and professional.

---

## 👨‍💻 Author

Built for GitHub portfolio presentation and Streamlit deployment.
