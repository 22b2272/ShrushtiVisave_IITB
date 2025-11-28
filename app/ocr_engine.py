import pytesseract
from PIL import Image
import io

class OCREngine:
    
    @staticmethod
    def extract_text(image_bytes: bytes) -> str:
        """Extract text using Tesseract OCR"""
        image = Image.open(io.BytesIO(image_bytes))
        
        # Configure Tesseract for better accuracy
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(image, config=custom_config)
        
        return text
