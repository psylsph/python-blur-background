from flask import Flask, render_template, request, send_file, jsonify
from PIL import Image, ImageFilter
import os
import io
from rembg import remove
import uuid
import base64

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)


def reduce_image_size(image, max_size=(800, 800)):
    """Reduces the image size while maintaining aspect ratio."""
    image.thumbnail(max_size)
    return image

def remove_background_from_image(image):
    """Removes background from the image."""
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes.seek(0)
    output_bytes = remove(image_bytes.read())
    output_image = Image.open(io.BytesIO(output_bytes))
    return output_image
    
def blur_image(image, blur_amount):
    """Blurs the image."""
    return image.filter(ImageFilter.GaussianBlur(radius=blur_amount))

def combine_images(foreground_image, background_image):
    """Combines the foreground (background removed) onto the background (blurred)."""
    background_image.paste(foreground_image, (0, 0), foreground_image)
    return background_image

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No image file provided"

        image_file = request.files['image']
        if image_file.filename == '':
            return "No image selected"
        
        blur_amount = request.form.get('blur', type=int, default=5)
        
        if image_file:
            try:
                image = Image.open(image_file)
                
                # Save original image with a unique name
                file_extension = os.path.splitext(image_file.filename)[1]
                unique_filename = f'{uuid.uuid4()}{file_extension}'
                
                upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                image.save(upload_path)

                resized_image = reduce_image_size(image.copy())

                background_removed = remove_background_from_image(resized_image.copy())
                blurred_image = blur_image(resized_image.copy(), blur_amount)
                
                combined_image = combine_images(background_removed, blurred_image)
                
                output_filename = f'combined_{unique_filename}'
                output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
                combined_image.save(output_path)
                
                return render_template('image_preview.html', output_file=output_filename)
            
            except Exception as e:
             return f"Error processing image: {e}"
    return render_template('upload.html',blur=5)

    
@app.route('/preview', methods=['POST'])
def preview_blur():
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    blur_amount = request.form.get('blur', type=int, default=5)

    try:
        image=Image.open(image_file)
        resized_image = reduce_image_size(image.copy())

        background_removed = remove_background_from_image(resized_image.copy())
        blurred_image = blur_image(resized_image.copy(), int(blur_amount))
        combined_image = combine_images(background_removed, blurred_image)

        
        
        base64_image = image_to_base64(combined_image)

        return jsonify({'image': base64_image})
    except Exception as e:
        print(e)
        return jsonify({'error': 'Error processing image'}), 500
    
@app.route('/output/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename), as_attachment=True)

@app.route('/view/<filename>')
def view_file(filename):
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    return send_file(file_path)

@app.route('/favicon.ico')
def favicon():
  return ""


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)
