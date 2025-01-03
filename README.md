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

## Usage

### Running the Web Interface

```bash
streamlit run ai_blur_background.py
```

This will start the Streamlit web interface where you can:
1. Upload an image
2. Adjust the blur strength
3. Download the processed image
