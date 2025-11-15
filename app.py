import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io

# -------------------------------
# 1 PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Image Classifier - ResNet18", layout="centered")


st.title(" Smart Waste Classifier")
st.write("Upload an image to classify waste as **Organic** or **Recyclable**")

# -------------------------------
# 2 LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model(weights_path, num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model

# Load class names 
class_names = ['O', 'R']  

model = load_model("resnet18_finetuned.pth", num_classes=len(class_names))

# -------------------------------
# 3 IMAGE TRANSFORMS
# -------------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -------------------------------
# 4 PREDICTION FUNCTION
# -------------------------------
def predict(image):
    image_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_t)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        top_prob, top_idx = torch.max(probs, dim=0)
        predicted_class = class_names[top_idx]
    return predicted_class, top_prob.item(), probs

# -------------------------------
# 5 IMAGE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("🕐 Running inference...")
    pred_class, pred_conf, probs = predict(image)

    st.success(f"✅ **Prediction:** {pred_class} ({pred_conf*100:.2f}% confidence)")
    st.write("Class probabilities:")
    prob_dict = {class_names[i]: float(probs[i]*100) for i in range(len(class_names))}
    st.bar_chart(prob_dict)

# -------------------------------
# 6 Grad-CAM VISUALIZATION
# -------------------------------
def gradcam(model, image_t):
    model.eval()
    target_layer = model.layer4[-1].conv2

    activations, gradients = [], []

    def forward_hook(_, __, output):
        activations.append(output)

    def backward_hook(_, grad_in, grad_out):
        gradients.append(grad_out[0])

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    outputs = model(image_t)
    pred_idx = outputs.argmax(dim=1)
    score = outputs[0, pred_idx]

    model.zero_grad()
    score.backward()

    act = activations[-1][0]
    grad = gradients[-1][0]
    weights = grad.mean(dim=(1, 2))

    cam = torch.relu((weights[:, None, None] * act).sum(dim=0))
    cam = (cam - cam.min()) / (cam.max() + 1e-8)

    h1.remove()
    h2.remove()

    return cam.detach().cpu().numpy(), pred_idx.item()

if uploaded_file is not None and st.checkbox("🔍 Show Grad-CAM Side-by-Side"):
    # Preprocess
    image_t = transform(image).unsqueeze(0)

    # Run grad-cam
    cam, pred_class = gradcam(model, image_t)

    # Make RGB image (0-1)
    img_np = np.array(image.resize((224, 224))) / 255.0

    # Create side-by-side figure
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Left: Original Image
    axs[0].imshow(img_np)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    # Right: Grad-CAM Heatmap
    axs[1].imshow(img_np)
    axs[1].imshow(cam, cmap="jet", alpha=0.45)
    axs[1].set_title(f"Grad-CAM (Pred: {pred_class})")
    axs[1].axis("off")

    st.pyplot(fig)


st.markdown(
    """
    <div style="text-align:center; padding-top: 20px; color: gray;">
    <hr>
    <p style="font-size:14px;">Built with ❤️ using PyTorch & Streamlit by ROYKEANE </p>
    </div>
    """,
    unsafe_allow_html=True
)
