
# Waste Image Classifier 
A deep learning image classification web application built using
- **PyTorch**(ResNet18 fine-tuned)
- **Streamlit**(web UI)
- **Grad-CAM**(explainability)
- **Render** (deployment)

The model predicts whether an uploaded image belongs to:
- **O**: Organic
- **R**: Recyclable


---

##  Demo
🔗 **Live App on Render:** [https://.onrender.com](#)  
🖼️ Example prediction interface:

<p align="center">
  <img src="assets/demo_gradcam.png" alt="Grad-CAM visualization" width="600"/>
</p>

---

##  Project Structure

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
- Purpose: Establish baseline performance

### Final Model/Deployed model: Fine-Tuned ResNet18
- Pretrained on ImageNet
- Fine-tuned last convolution block (`layer4`)
- Accuracy: **95% on test set**
- Explainability: **Grad-CAM** visualization

---

## Setup Instructions
### 1 Clone the repository
git clone https://github.com/Roy16Keane/Smart-waste-image-classifier.git
cd Smart-waste-image-classifier

### 2 Install dependencies
pip install -r requirements.txt

### 3 Run locally
streamlit run app.py
## Deployment on Render
### 1 Push your project to GitHub
Ensure your repo contains:
- app.py
- requirements.txt
- models/resnet18_finetuned.pth
### 2 Go to https://render.com
 → New Web Service
 ### 3 Set these commands
 Build command:
 pip install -r requirements.txt

Start command:
streamlit run app.py

 


## Upload → Predict → Explain
The app allows users to:
- Upload any image
- See the predicted class + confidence score
- Visualize a Grad-CAM heatmap showing what the model focused on
- Compare predictions between classes
  
The interface is fast, simple, and beginner 

## Features
**Real-time predictions:**
Fast inference using PyTorch's optimized ResNet18

**Interactive Grad-CAM:**
See exactly what part of the image influenced the classification.

**Clean UI:**
Custom CSS styling for a professional look

**Deployment-ready:**
Designed to run effortlessly on Render or Streamlit Cloud

**Fully reproductible:**
All dependencies explicitly defined

## Future Enhancement
Add more classes (plastic, metal, glass, paper)

Convert model to ONNX for mobile 

Add multi-class Grad-CAM 

Deploy FastAPI backend + Streamlit frontend

Build auto-retraining MLOps pipeline







