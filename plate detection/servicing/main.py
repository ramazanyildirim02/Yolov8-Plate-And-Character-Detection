# Libraries
import cv2
from PIL import Image
import streamlit as st
from helper import detect_plate

# Title
st.title("🚘 Plate Recognition System 🚘")

# Header
st.header("Upload an Image")

# Files
file = st.file_uploader("", type= ["png", "jpg", "jpeg"])

# Model
model_path = "models/plate_detection.pt"

# Images
if file is not None:
    # Original Image
    st.header("Original Image")
    image = Image.open(file).convert("RGB")
    st.image(image, use_container_width= True)

    # Processed Image
    st.header("Detection Result")
    detection_result, cropped_image, is_detected = detect_plate(image, model_path)

    if is_detected is not 0:
        st.image(detection_result, use_container_width= True)
        st.image(cropped_image, use_container_width= True)
        st.write("### [INFO]... Plate is detected !")

    else:
        st.image(detection_result, use_container_width= True)
        st.write("### [INFO]... Plate is not detected !")
  


