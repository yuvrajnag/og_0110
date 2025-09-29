import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from model import create_student

# Load model
ckpt = torch.load("c:/Users/Ashwith/OneDrive/Desktop/zoolake project/checkpoints/best_student.pth", map_location="cpu")
classes = ckpt['classes']
model = create_student(len(classes))
model.load_state_dict(ckpt['model_state'])
model.eval()

# Transform
tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

st.title("ZooLake Plankton Classifier 🐠")
st.write("Upload a plankton crop image and I’ll predict the species.")

uploaded = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)

    x = tf(img).unsqueeze(0)

    with torch.no_grad():
        pred = model(x).argmax(1).item()

    st.success(f"Prediction: **{classes[pred]}**")
