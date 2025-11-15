
# Smart Waste Image Classifier using Transfer Learning (ResNet18 + Streamlit)

A deep learning web app that classifies uploaded waste images into **Organic (O)** or **Recyclable (R)** categories, powered by **PyTorch** and **Streamlit**.  
This project demonstrates **transfer learning**, **Grad-CAM explainability**, and full deployment through **Render** for public access.

---

##  Demo
🔗 **Live App on Render:** [https://.onrender.com](#)  
🖼️ Example prediction interface:

<p align="center">
  <img src="assets/demo_gradcam.png" alt="Grad-CAM visualization" width="600"/>
</p>

---

## 📂 Project Structure
resnet18-classifier/
│
├── app.py # Streamlit app entry point
├── requirements.txt # Dependencies for Render deployment
├── README.md # Project documentation
│
├── models/
│ └── resnet18_finetuned.pth # Fine-tuned model weights
│
├── utils/
│ └── gradcam_utils.py # Grad-CAM visualization helper functions
│
└── assets/
├── demo_gradcam.png # Example Grad-CAM output
└── sample_O.jpg # Example image (Organic)


---

##  Model Overview

###  Baseline: Custom SimpleCNN
- Trained from scratch
- Accuracy: **89% on testing set**
- Purpose: Educational baseline

### Final Model/Deployed model: Fine-Tuned ResNet18
- Pretrained on ImageNet
- Fine-tuned last convolution block (`layer4`)
- Accuracy: **95% on test set**
- Explainability: **Grad-CAM** visualization

---

## Setup Instructions

### 1 Clone the repository
```bash
git clone https://github.com/<Roy16Keane>/Smart-waste-image-classifier.git
cd resnet18-classifier



### 2 Install dependencies
```bash
pip install -r requirements.txt


### 3 Run locally
```bash
streamlit run app.py

