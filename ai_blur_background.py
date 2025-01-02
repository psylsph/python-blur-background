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

def remove_background_from_image(image: PILImage, model_name="u2net"):
    """Removes background from the image."""
    with tempfile.TemporaryDirectory() as tmp:
        source_image = os.path.join(tmp, 'source_image.png')
        image.save(source_image)
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

def disable_remove_background():
    st.session_state["remove_background"] = False

def enable_remove_background():
    st.session_state["remove_background"] = True

st.set_page_config(page_title="AI Blur Background", page_icon=":camera:", layout="centered")
st.html("""
    <style>
        .stMainBlockContainer {
            max-width:21rem;
        }
    </style>
    """
)

if "remove_background" not in st.session_state:
    st.session_state["remove_background"] = True

uploaded_file = st.file_uploader("**Select an image...**", type=["jpg", "jpeg", "png", "heic"], on_change=enable_remove_background)
model_name = st.selectbox("**Select Model**", ["bria-rmbg", "u2net"], index=1, on_change=enable_remove_background)
blur_amount = st.slider("**Blur Amount**", 0, 50, 25, on_change=disable_remove_background)

if uploaded_file is not None:

    with WandImage( blob=uploaded_file.getvalue()) as wand_image:
        
        preview_width = 300

        img_buffer = numpy.asarray(bytearray(wand_image.make_blob(format='png')), dtype='uint8')
        bytesio = io.BytesIO(img_buffer)
        image = PILImage.open(bytesio)
        st.image(image, caption="Uploaded Image", width=preview_width)
        
        if st.session_state["remove_background"]:
            with st.spinner("Removing background..."):
                st.session_state["background_removed"] = remove_background_from_image(image.copy(), model_name=model_name)
                st.image(st.session_state["background_removed"], caption="Background Removed", width=preview_width)

        blurred_image = blur_image(image.copy(), blur_amount)

        with st.spinner("Combining images..."):
            combined_image = combine_images(st.session_state["background_removed"], blurred_image)
            st.image(combined_image, caption="Combined Image", width=preview_width)

        combined_image.convert('RGB').save('combined_image.jpg')
        
        st.session_state["remove_background"] = False
        
        with open('combined_image.jpg', 'rb') as f:
            st.download_button('Download File', f, file_name='combined_image.jpg', mime='image/jpeg')