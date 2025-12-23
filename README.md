# 🔬 DermaAI - Smart Skin Disease Detection

![Status](https://img.shields.io/badge/Status-Production-success)
![Tech Stack](https://img.shields.io/badge/Stack-FastAPI%20|%20TensorFlow%20|%20JS-blue)
![Mobile](https://img.shields.io/badge/Mobile-Camera%20Ready-orange)

**DermaAI** is an end-to-end medical AI application capable of classifying **12 different skin diseases**. It uses a custom-trained **EfficientNetB1** model and features a mobile-responsive UI with real-time camera integration.

🔗 **Live Demo:** [PASTE YOUR VERCEL LINK HERE]

---

## 🚀 Key Features
* **Multi-Class Diagnosis:** Detects 12 conditions including Acne, Melanoma, Eczema, and Psoriasis.
* **Advanced AI Model:** Powered by **EfficientNetB1** using Transfer Learning, trained on a merged dataset of ~16,000 clinical and dermoscopic images.
* **Imbalance Handling:** Implements automated **Class Weights** to accurately detect rare diseases (e.g., Cancer) despite dataset imbalance.
* **Mobile-First Design:** Built-in camera support allows users to analyze skin lesions directly from their phone.
* **Smart Analysis:** Provides a top prediction along with "Second Best Guesses" and confidence scores.

---

## 🛠️ Tech Stack
* **Frontend:** HTML5, Tailwind CSS, Vanilla JavaScript (Hosted on **Vercel**)
* **Backend:** Python 3.10, FastAPI, Uvicorn (Hosted on **Render**)
* **Machine Learning:** TensorFlow/Keras, NumPy, Pillow, EfficientNetB1

---

## 🏥 Diseases Detected
The model is trained to recognize the following 12 classes:
1.  **Acne**
2.  **Actinic Keratoses** (Pre-cancerous)
3.  **Basal Cell Carcinoma** (Cancer)
4.  **Benign Keratosis**
5.  **Dermatofibroma**
6.  **Eczema**
7.  **Melanocytic Nevi** (Mole)
8.  **Melanoma** (Cancer)
9.  **Nail Fungus**
10. **Psoriasis**
11. **Ringworm** (Fungal)
12. **Vascular Lesions**

---

## 📸 Screenshots

| Desktop View | |
|:---:|:---:|
| ![Desktop]<img width="2940" height="1500" alt="image" src="https://github.com/user-attachments/assets/7300f52c-09a6-4ace-a108-cc5b6ee570a2" />


---

## ⚙️ How to Run Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/DermaAI.git](https://github.com/YOUR_USERNAME/DermaAI.git)
cd DermaAI
