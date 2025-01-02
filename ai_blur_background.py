import cv2
import tempfile
from wand.image import Image as WandImage
from PIL import Image as PILImage
from PIL import ImageFilter
import numpy
import os
import io
from rembg import new_session, remove
import base64
import streamlit as st


def remove_background_from_image(image: PILImage):
    """Removes background from the image."""
    with tempfile.TemporaryDirectory() as tmp:
        source_image = os.path.join(tmp, 'source_image.png')
        image.save(source_image)
        model_name = "isnet-general-use"
        session = new_session(model_name)
        output = remove(PILImage.open(source_image), session=session)
    return output
    
def blur_image(image: PILImage, blur_amount):
    """Blurs the image."""
    return image.filter(ImageFilter.GaussianBlur(radius=blur_amount))

def combine_images(foreground_image, background_image):
    """Combines the foreground (background removed) onto the background (blurred)."""
    background_image.paste(foreground_image, (0, 0), foreground_image)
    return background_image

def image_to_base64(image: PILImage):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


st.set_page_config(page_title="AI Blur Background", page_icon=":camera:")

uploaded_file = st.file_uploader("**Select an image...**", type=["jpg", "jpeg", "png", "heic"])
blur_amount = st.slider("**Blur Amount**", 0, 50, 25)
if uploaded_file is not None:
    with WandImage( blob=uploaded_file.getvalue()) as wand_image:
        img_buffer = numpy.asarray(bytearray(wand_image.make_blob(format='png')), dtype='uint8')
        bytesio = io.BytesIO(img_buffer)
        image = PILImage.open(bytesio)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Removing background..."):
            background_removed = remove_background_from_image(image.copy())
            st.image(background_removed, caption="Background Removed", use_container_width=True)
        blurred_image = blur_image(image.copy(), blur_amount)
        with st.spinner("Combining images..."):
            combined_image = combine_images(background_removed, blurred_image)
            st.image(combined_image, caption="Combined Image", use_container_width=True)
        combined_image.convert('RGB').save('combined_image.jpg')
        with open('combined_image.jpg', 'rb') as f:
            st.download_button('Download File', f, file_name='combined_image.jpg', mime='image/jpeg')