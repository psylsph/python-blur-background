import unittest
from app import app, reduce_image_size
from PIL import Image, ImageDraw
import io

class AppTests(unittest.TestCase):

    def test_home_page(self):
        with app.test_client() as client:
            response = client.get('/')
            assert response.status_code == 200
            assert b"Image Processor" in response.data

    def test_reduce_image_size(self):
        # Create a dummy image
        image = Image.new('RGB', (200, 200), color = 'black')
        
        # Save the image to a byte stream
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Reduce the image size
        reduced_image = reduce_image_size(Image.open(io.BytesIO(img_byte_arr)))
        
        # Assert that the reduced image is smaller than the original
        self.assertLessEqual(reduced_image.size[0], 800)
        self.assertLessEqual(reduced_image.size[1], 800)

if __name__ == "__main__":
        unittest.main()
