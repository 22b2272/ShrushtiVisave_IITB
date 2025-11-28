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

logger = logging.getLogger(__name__)


class DocumentUtils:
    """Utilities for document handling"""
    
    @staticmethod
    def download_document(url: str, timeout: int = 30) -> bytes:
        """
        Download document from URL with retry logic
        
        Args:
            url: Document URL
            timeout: Request timeout in seconds
            
        Returns:
            Document bytes
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=timeout)
                response.raise_for_status()
                logger.info(f"Successfully downloaded document ({len(response.content)} bytes)")
                return response.content
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Download attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to download document after {max_retries} attempts")
                    raise
    
    @staticmethod
    def detect_document_type(content: bytes) -> str:
        """
        Detect if document is PDF or image
        
        Args:
            content: Document bytes
            
        Returns:
            'pdf' or 'image'
        """
        # Check PDF magic number
        if content[:4] == b'%PDF':
            return 'pdf'
        
        # Check common image magic numbers
        image_signatures = {
            b'\xFF\xD8\xFF': 'jpeg',
            b'\x89PNG\r\n\x1a\n': 'png',
            b'GIF87a': 'gif',
            b'GIF89a': 'gif',
            b'II*\x00': 'tiff',
            b'MM\x00*': 'tiff',
        }
        
        for signature in image_signatures:
            if content.startswith(signature):
                return 'image'
        
        raise ValueError("Unsupported document type")
    
    @staticmethod
    def pdf_to_images(pdf_bytes: bytes, dpi: int = 300) -> List[bytes]:
        """
        Convert PDF pages to images
        
        Args:
            pdf_bytes: PDF document bytes
            dpi: Resolution for conversion
            
        Returns:
            List of image bytes (one per page)
        """
        try:
            images = convert_from_bytes(pdf_bytes, dpi=dpi)
            image_bytes_list = []
            
            for i, image in enumerate(images):
                # Convert PIL Image to bytes
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                image_bytes_list.append(img_byte_arr.read())
                logger.info(f"Converted PDF page {i + 1} to image")
            
            return image_bytes_list
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            raise
    
    @staticmethod
    def validate_image(image_bytes: bytes) -> bool:
        """
        Validate if bytes represent a valid image
        
        Args:
            image_bytes: Image bytes
            
        Returns:
            True if valid, False otherwise
        """
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.verify()
            return True
        except Exception:
            return False
    
    @staticmethod
    def get_image_dimensions(image_bytes: bytes) -> Tuple[int, int]:
        """
        Get image dimensions
        
        Args:
            image_bytes: Image bytes
            
        Returns:
            (width, height) tuple
        """
        image = Image.open(io.BytesIO(image_bytes))
        return image.size
    
    @staticmethod
    def calculate_document_hash(content: bytes) -> str:
        """
        Calculate hash of document for caching/deduplication
        
        Args:
            content: Document bytes
            
        Returns:
            SHA256 hash string
        """
        return hashlib.sha256(content).hexdigest()


class TextUtils:
    """Utilities for text processing"""
    
    @staticmethod
    def clean_ocr_text(text: str) -> str:
        """
        Clean OCR text by removing excessive whitespace and special characters
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    @staticmethod
    def extract_amount(text: str) -> Optional[float]:
        """
        Extract amount from text string
        
        Args:
            text: Text containing amount
            
        Returns:
            Extracted amount or None
        """
        # Pattern to match amounts: 1234.56, 1,234.56, ₹1234.56
        pattern = r'[₹$]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        match = re.search(pattern, text)
        
        if match:
            amount_str = match.group(1).replace(',', '')
            try:
                return float(amount_str)
            except ValueError:
                return None
        return None
    
    @staticmethod
    def normalize_item_name(name: str) -> str:
        """
        Normalize item name while preserving original format
        
        Args:
            name: Item name
            
        Returns:
            Normalized name
        """
        # Remove excessive whitespace but preserve structure
        name = re.sub(r'\s+', ' ', name)
        return name.strip()


class ValidationUtils:
    """Utilities for validation"""
    
    @staticmethod
    def validate_amount_calculation(
        rate: float, 
        quantity: float, 
        amount: float, 
        tolerance: float = 0.01
    ) -> bool:
        """
        Validate if amount = rate × quantity (within tolerance)
        
        Args:
            rate: Item rate
            quantity: Item quantity
            amount: Item amount
            tolerance: Acceptable difference
            
        Returns:
            True if valid, False otherwise
        """
        expected = rate * quantity
        return abs(expected - amount) <= tolerance
    
    @staticmethod
    def check_duplicate_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find duplicate items in a list
        
        Args:
            items: List of item dictionaries
            
        Returns:
            List of duplicate items
        """
        seen = {}
        duplicates = []
        
        for item in items:
            key = (item.get('item_name'), item.get('item_amount'))
            if key in seen:
                duplicates.append(item)
            else:
                seen[key] = item
        
        return duplicates
    
    @staticmethod
    def calculate_total(items: List[Dict[str, Any]]) -> float:
        """
        Calculate total from list of items
        
        Args:
            items: List of item dictionaries with 'item_amount' field
            
        Returns:
            Total amount
        """
        return sum(item.get('item_amount', 0.0) for item in items)


class PerformanceUtils:
    """Utilities for performance monitoring"""
    
    @staticmethod
    def timer(func):
        """
        Decorator to measure function execution time
        
        Usage:
            @PerformanceUtils.timer
            def my_function():
                pass
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.info(f"{func.__name__} executed in {end_time - start_time:.2f}s")
            return result
        return wrapper
    
    @staticmethod
    def log_memory_usage():
        """Log current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            logger.info(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")
        except ImportError:
            logger.warning("psutil not installed, cannot log memory usage")


class ErrorUtils:
    """Utilities for error handling"""
    
    @staticmethod
    def safe_float_conversion(value: Any, default: float = 0.0) -> float:
        """
        Safely convert value to float
        
        Args:
            value: Value to convert
            default: Default value if conversion fails
            
        Returns:
            Float value
        """
        try:
            if isinstance(value, str):
                # Remove currency symbols and commas
                value = re.sub(r'[₹$,]', '', value)
            return float(value)
        except (ValueError, TypeError):
            logger.warning(f"Failed to convert {value} to float, using default {default}")
            return default
    
    @staticmethod
    def safe_int_conversion(value: Any, default: int = 0) -> int:
        """
        Safely convert value to int
        
        Args:
            value: Value to convert
            default: Default value if conversion fails
            
        Returns:
            Int value
        """
        try:
            return int(value)
        except (ValueError, TypeError):
            logger.warning(f"Failed to convert {value} to int, using default {default}")
            return default
    
    @staticmethod
    def format_error_response(error: Exception) -> Dict[str, Any]:
        """
        Format error for API response
        
        Args:
            error: Exception object
            
        Returns:
            Error response dictionary
        """
        return {
            "is_success": False,
            "error": {
                "type": type(error).__name__,
                "message": str(error)
            },
            "token_usage": {
                "total_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0
            },
            "data": {
                "pagewise_line_items": [],
                "total_item_count": 0
            }
        }


class CacheUtils:
    """Simple in-memory cache for frequently accessed data"""
    
    _cache = {}
    
    @classmethod
    def get(cls, key: str) -> Optional[Any]:
        """Get value from cache"""
        return cls._cache.get(key)
    
    @classmethod
    def set(cls, key: str, value: Any, ttl: int = 3600):
        """
        Set value in cache with TTL
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (not implemented, placeholder)
        """
        cls._cache[key] = value
    
    @classmethod
    def clear(cls):
        """Clear all cache"""
        cls._cache.clear()


# Constants
SUPPORTED_EXTENSIONS = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.gif']
MAX_FILE_SIZE_MB = 50
DEFAULT_DPI = 300
OCR_CONFIDENCE_THRESHOLD = 0.6

# Regex patterns
AMOUNT_PATTERN = r'[₹$]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
PHONE_PATTERN = r'\+?\d{10,15}'
EMAIL_PATTERN = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'


def format_currency(amount: float, currency: str = '₹') -> str:
    """
    Format amount as currency string
    
    Args:
        amount: Amount to format
        currency: Currency symbol
        
    Returns:
        Formatted string
    """
    return f"{currency}{amount:,.2f}"


def truncate_string(text: str, max_length: int = 100) -> str:
    """
    Truncate string to max length
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def is_valid_url(url: str) -> bool:
    """
    Check if string is a valid URL
    
    Args:
        url: URL string
        
    Returns:
        True if valid URL, False otherwise
    """
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return url_pattern.match(url) is not None
