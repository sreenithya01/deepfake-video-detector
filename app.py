
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from torchvision import models
import cv2
import tempfile
import numpy as np
# Detect if GPU is available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the trained model (MobileNetV2)
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load('deepfake_model.pth', map_location=device))
model.eval()
model.to(device)


# Image transformation (must be same as training!)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])



st.write("---")
st.header("🎥 Upload Video for Deepfake Detection")
video_file = st.file_uploader("Upload a video...", type=["mp4", "mov", "avi"])

if video_file is not None:
    # Save uploaded video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name

    # Show video in Streamlit
    st.video(video_file)

    # Load video using OpenCV
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    st.write(f"Total Frames in Video: {frame_count}")

    results = []
    sampled_frames = 0

    st.write("Analyzing video frames...")

    while True:
        ret, frame = cap.read()

        if not ret or sampled_frames >= 10:  # Limit to 10 frames for speed
            break

        # Convert OpenCV BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        # Transform frame
        img_tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)
            confidence = probs[0][1].item()
            predicted = torch.argmax(probs, 1).item()

        label = 'Fake' if predicted == 1 else 'Real'
        results.append(label)

        st.image(pil_img, caption=f"Frame {sampled_frames+1}: {label} ({confidence*100:.2f}% Fake)", use_column_width=True)
        sampled_frames += 1

    cap.release()

    # Final decision
    fake_count = results.count('Fake')
    real_count = results.count('Real')

    st.write("---")
    if fake_count > real_count:
        st.subheader(f"Final Video Prediction: 🔴 Fake ({fake_count}/10 frames)")
    else:
        st.subheader(f"Final Video Prediction: 🟢 Real ({real_count}/10 frames)")
