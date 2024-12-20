# Image Blur Background App

This application allows you to upload an image, blur the background, and download the processed image.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd python-blur-background
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/macOS
   venv\Scripts\activate  # On Windows
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   mkdir ai-model
   curl -L https://huggingface.co/stoned0651/isnet_dis.onnx/resolve/main/isnet_dis.onnx -O ai-model/isnet_dis.onnx # On Linux/macOS
   curl https://huggingface.co/stoned0651/isnet_dis.onnx/resolve/main/isnet_dis.onnx -O ai-model/isnet_dis.onnx # On Windows
   ```

## Building

The application is built using Flask and Python. No specific build steps are required.

## Testing

To run the tests, use the following command:

```bash
python -m unittest tests/test_app.py
```

## Running via Docker

1. Build the Docker image:
   ```bash
   docker build -t image-blur-app .
   ```

2. Run the Docker container:
   ```bash
   docker run -p 5000:5000 image-blur-app
   ```

   The application will be available at `http://localhost:5000`.
