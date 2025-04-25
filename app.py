import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import tempfile
import sys
import json
import uuid
from pathlib import Path

# Add local yolov5 path
sys.path.append(str(Path(__file__).resolve().parent / "yolov5"))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.augmentations import letterbox

# Load custom CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles.css")

st.markdown(
    "<h1 style='text-align: center; font-size: 40px; margin-top: 0;'>Real-Time Object Detection with YOLO</h1>",
    unsafe_allow_html=True
)

st.markdown("Upload an image or use your webcam to detect people, objects, and more.")

# Mode selector
mode = st.radio("Select Input Mode:", ["Upload Image", "Webcam"], horizontal=True)

confidence_threshold = 0.25

# Load model
device = select_device("cpu")
model = DetectMultiBackend("yolov5m.pt", device=device)
model.warmup(imgsz=(1, 3, 640, 640))

def detect_image(image_np, conf_thres):
    if isinstance(image_np, Image.Image):
        image_np = np.array(image_np)

    if not isinstance(image_np, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    if image_np.dtype != np.uint8:
        image_np = image_np.astype(np.uint8)

    img = letterbox(image_np, 640, stride=32, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)

    im_tensor = torch.from_numpy(img).to(device).float()
    im_tensor /= 255.0
    if im_tensor.ndimension() == 3:
        im_tensor = im_tensor.unsqueeze(0)

    pred = model(im_tensor)
    pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=0.45)[0]

    results = []
    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(im_tensor.shape[2:], pred[:, :4], image_np.shape).round()
        for *xyxy, conf, cls in pred:
            label = f'{model.names[int(cls)]} {conf:.2f}'
            results.append({"label": label, "box": [float(x.item()) for x in xyxy]})
            cv2.rectangle(image_np, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
            cv2.putText(image_np, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return image_np, results

# Upload Image
if mode == "Upload Image":
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if file:
        img = Image.open(file)
        img_np = np.array(img)
        result_img, detections = detect_image(img_np, confidence_threshold)
        st.image(result_img, caption="🖼️ Detected Image", use_container_width=True)
        st.subheader("📝 Detection Results")
        st.json(detections)

        with open("upload_detections.json", "w") as jf:
            json.dump(detections, jf, indent=2)
        with open("upload_detections.json", "rb") as jf:
            st.download_button(
                label="📥 Download Detection JSON",
                data=jf,
                file_name="upload_detections.json",
                mime="application/json",
                key="upload-json"
            )

# Webcam
else:
    st.warning("Click the 🛑 Stop Webcam button to stop live detection.")
    cap = cv2.VideoCapture(0)
    stop = st.button("🛑 Stop Webcam")
    stframe = st.empty()

    if cap.isOpened():
        st.success("✅ Webcam started successfully!")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Webcam error! Frame not captured.")
                break

            img_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result_img, detections = detect_image(img_np, confidence_threshold)
            stframe.image(result_img, channels="RGB", use_container_width=True)

            if stop:
                cap.release()

                st.subheader("📝 Detection Results from Webcam")
                st.json(detections)

                with open("webcam_detections.json", "w") as jf:
                    json.dump(detections, jf, indent=2)
                with open("webcam_detections.json", "rb") as jf:
                    st.download_button(
                        label="📥 Download Webcam Detection JSON",
                        data=jf,
                        file_name="webcam_detections.json",
                        mime="application/json",
                        key=f"download-json-{uuid.uuid4()}"
                    )
                break

st.markdown('<p style="text-align:center; color:gray;">© 2025 Harpreet Kaur</p>', unsafe_allow_html=True)
