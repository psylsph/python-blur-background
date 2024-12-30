import cv2
import tempfile
from PIL import Image, ImageFilter
import os
import requests
from os.path import isfile
import io
from dis_bg_remover import remove_background
import base64
import streamlit as st

MODEL_PATH = 'isnet_dis.onnx'

def download_model(url):
    response = requests.get(url)
    with open(MODEL_PATH, mode="wb") as file:
        file.write(response.content)
    print(f"Downloaded file {MODEL_PATH}")

def remove_background_from_image(image: Image):
    """Removes background from the image."""
    with tempfile.TemporaryDirectory() as tmp:
        source_image = os.path.join(tmp, 'source_image.png')
        bg_removed_image = os.path.join(tempfile.gettempdir(), 'bg_removed_image.png')
        image.save(source_image, format='PNG')
        extracted_img, mask = remove_background(MODEL_PATH, source_image)
        cv2.imwrite(bg_removed_image, extracted_img)
        output_image = Image.open(bg_removed_image)
    return output_image
    
def blur_image(image: Image, blur_amount):
    """Blurs the image."""
    return image.filter(ImageFilter.GaussianBlur(radius=blur_amount))

def combine_images(foreground_image, background_image):
    """Combines the foreground (background removed) onto the background (blurred)."""
    background_image.paste(foreground_image, (0, 0), foreground_image)
    return background_image

def image_to_base64(image: Image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


st.set_page_config(layout="wide")

if not isfile(MODEL_PATH):
    with st.spinner("Downloading model..."):
        download_model("https://huggingface.co/stoned0651/isnet_dis.onnx/resolve/main/isnet_dis.onnx")
        st.write("Model downloaded.")

uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "heic"])
blur_amount = st.sidebar.slider("Blur Amount", 0, 20, 10)
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("")
    st.write("Removing background...")
    background_removed = remove_background_from_image(image.copy())
    st.image(background_removed, caption="Background Removed", use_container_width=True)
    st.write("")
    st.write("Blurring background...")
    blurred_image = blur_image(image.copy(), blur_amount)
    st.image(blurred_image, caption="Blurred Image", use_container_width=True)
    st.write("")
    st.write("Combining images...")
    combined_image = combine_images(background_removed, blurred_image)
    st.image(combined_image, caption="Combined Image", use_container_width=True)
    st.write("")
    st.write("Done!")