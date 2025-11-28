import cv2
import numpy as np
from PIL import Image
import io

class DocumentPreprocessor:
    
    @staticmethod
    def enhance_image(image_bytes: bytes) -> bytes:
        """Enhance image quality for better OCR"""
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Adaptive thresholding for better text clarity
        binary = cv2.adaptiveThreshold(
            denoised, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Deskew if needed
        coords = np.column_stack(np.where(binary > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        if abs(angle) > 0.5:  # Only rotate if needed
            (h, w) = binary.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            binary = cv2.warpAffine(
                binary, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
        
        # Convert back to bytes
        _, buffer = cv2.imencode('.png', binary)
        return buffer.tobytes()
    
    @staticmethod
    def detect_fraud(image_bytes: bytes) -> List[str]:
        """Detect potential fraud indicators"""
        fraud_flags = []
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detect white patches (whitener)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
        white_ratio = np.sum(white_mask > 0) / white_mask.size
        
        if white_ratio > 0.15:  # More than 15% white
            fraud_flags.append("Excessive white patches detected")
        
        # Detect font inconsistencies (simplified)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 1000:  # Too many irregular contours
            fraud_flags.append("Potential font inconsistency")
        
        return fraud_flags
