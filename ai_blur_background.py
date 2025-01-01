from wand.image import Image as WandImage
from PIL import Image as PILImage
from PIL import ImageFilter
import numpy
import io
import base64
import streamlit as st
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

def remove_background_from_image(image: PILImage):
    """Removes background from the image."""
    model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
    torch.set_float32_matmul_precision(['high', 'highest'][0])
    model.to('cpu')
    model.eval()
   

    # Data settings
    image_size = (1024, 1024)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    input_images = transform_image(image).unsqueeze(0).to('cpu')

    # Prediction
    with torch.no_grad():
        preds = model(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    image.putalpha(mask)
    return image
    
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


st.set_page_config(page_title="AI Blur Background App", page_icon=":material/star_half:", initial_sidebar_state="expanded")
st.markdown("#### AI Blur Background App")

uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "heic"])
blur_amount = st.sidebar.slider("Blur Amount", 0, 50, 25)
if uploaded_file is not None:
    with WandImage( blob=uploaded_file.getvalue()) as wand_image:
        img_buffer = numpy.asarray(bytearray(wand_image.make_blob(format='png')), dtype='uint8')
        bytesio = io.BytesIO(img_buffer)
        image = PILImage.open(bytesio)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.write("")
        with st.spinner("Removing background..."):
            background_removed = remove_background_from_image(image.copy())
            st.image(background_removed, caption="Background Removed", use_container_width=True)
        blurred_image = blur_image(image.copy(), blur_amount)
        with st.spinner("Combining images..."):
            combined_image = combine_images(background_removed, blurred_image)
            st.image(combined_image, caption="Final Image", use_container_width=True)
        st.write("")
        combined_image.convert('RGB').save('blur_background_image.jpg')
        with open('blur_background_image.jpg', 'rb') as f:
            st.download_button('Download File', f, file_name='blur_background_image.jpg', mime='image/jpeg')