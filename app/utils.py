"""
Utility functions for bill extraction API
"""

import io
import logging
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import requests
from pdf2image import convert_from_bytes
import time
from functools import wraps
import numpy as np
import cv2
import pytesseract

logger = logging.getLogger(__name__)


# ===============================================================
# DOCUMENT UTILITIES
# ===============================================================

class DocumentUtils:
    """Utilities for document handling"""

    @staticmethod
    def download_document(url: str, timeout: int = 30) -> bytes:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=timeout)
                response.raise_for_status()
                return response.content
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

    @staticmethod
    def detect_document_type(content: bytes) -> str:
        if content[:4] == b'%PDF':
            return 'pdf'
        signatures = {
            b'\xFF\xD8\xFF': 'jpeg',
            b'\x89PNG\r\n\x1a\n': 'png',
            b'GIF87a': 'gif',
            b'GIF89a': 'gif'
        }
        for sig in signatures:
            if content.startswith(sig):
                return 'image'
        raise ValueError("Unsupported document type")

    @staticmethod
    def pdf_to_images(pdf_bytes: bytes, dpi: int = 300) -> List[bytes]:
        images = convert_from_bytes(pdf_bytes, dpi=dpi)
        page_bytes = []
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            buf.seek(0)
            page_bytes.append(buf.read())
        return page_bytes


# ===============================================================
# OCR UTILITIES
# ===============================================================

class OCRUtils:
    """Utilities for OCR preprocessing and extraction"""

    @staticmethod
    def preprocess_image(pil_image: Image.Image) -> Image.Image:
        img = np.array(pil_image.convert('L'))
        _, img_thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
        return Image.fromarray(img_thresh)

    @staticmethod
    def extract_line_items_from_text(text: str) -> (list, float):
        line_items = []
        for line in text.split("\n"):
            line = line.strip()
            match = re.search(r"(.+?)\s+([\d,]+\.\d{2})$", line)
            if match:
                item = match.group(1)
                amount = float(match.group(2).replace(",", ""))
                line_items.append({"item_name": item, "item_amount": amount})
        subtotal = sum([x["item_amount"] for x in line_items])
        return line_items, subtotal


# ===============================================================
# TEXT UTILITIES
# ===============================================================

class TextUtils:
    @staticmethod
    def clean_ocr_text(text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


# ===============================================================
# ERROR UTILITIES
# ===============================================================

class ErrorUtils:
    @staticmethod
    def safe_float_conversion(value: Any, default: float = 0.0):
        try:
            if isinstance(value, str):
                value = re.sub(r'[₹$,]', '', value)
            return float(value)
        except:
            return default

    @staticmethod
    def format_error_response(error: Exception) -> Dict[str, Any]:
        return {
            "is_success": False,
            "error": {"type": type(error).__name__, "message": str(error)},
            "data": {"pagewise_line_items": [], "total_item_count": 0}
        }
