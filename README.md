# Image Blur Background App

This Streamlit application allows you to upload an image and automatically blur its background using AI segmentation.

## Requirements

- Python 3.8 or higher
- 2GB of free disk space for AI model
- At least 4GB RAM recommended

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/stuarth/python-blur-background.git
   cd python-blur-background
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/macOS
   venv\Scripts\activate  # On Windows
   ```

3. Install dependencies:
   ```bash
   sudo apt-get update
   sudo apt-get install -y libgl1-mesa-glx libmagickwand-dev
   pip install -r requirements.txt
   ```

4. Download the AI model:
   ```bash
   # On Linux/macOS
   curl -L https://huggingface.co/stoned0651/isnet_dis.onnx/resolve/main/isnet_dis.onnx -o isnet_dis.onnx
   # On Windows
   curl https://huggingface.co/stoned0651/isnet_dis.onnx/resolve/main/isnet_dis.onnx -o isnet_dis.onnx
   ```

## Usage

### Running the Web Interface

```bash
streamlit run app.py
```

This will start the Streamlit web interface where you can:
1. Upload an image
2. Adjust the blur strength
3. Download the processed image
