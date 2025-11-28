"""
Bill Extraction API
Medical bill data extraction with fraud detection
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

import logging
import sys
from typing import Dict, Any

# Configure logging for the entire application
def setup_logging(log_level: str = "INFO") -> None:
    """
    Setup application logging
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('bill_extraction.log')
        ]
    )

# Setup logging when module is imported
setup_logging()

logger = logging.getLogger(__name__)
logger.info(f"Bill Extraction API v{__version__} initialized")

# Import key components for easier access
from .models import (
    BillExtractionRequest,
    BillExtractionResponse,
    BillItem,
    PagewiseLineItems,
    TokenUsage,
    BillData
)

from .utils import (
    DocumentUtils,
    TextUtils,
    ValidationUtils,
    PerformanceUtils,
    ErrorUtils,
    format_currency,
    is_valid_url
)

# Export main components
__all__ = [
    # Models
    'BillExtractionRequest',
    'BillExtractionResponse',
    'BillItem',
    'PagewiseLineItems',
    'TokenUsage',
    'BillData',
    
    # Utils
    'DocumentUtils',
    'TextUtils',
    'ValidationUtils',
    'PerformanceUtils',
    'ErrorUtils',
    'format_currency',
    'is_valid_url',
    
    # Version info
    '__version__',
    '__author__',
]


def get_app_info() -> Dict[str, Any]:
    """
    Get application information
    
    Returns:
        Dictionary with app info
    """
    return {
        "name": "Bill Extraction API",
        "version": __version__,
        "author": __author__,
        "description": "Medical bill data extraction with fraud detection",
        "endpoints": [
            "/extract-bill-data",
            "/health",
            "/docs"
        ]
    }


# Configuration defaults
class Config:
    """Application configuration"""
    
    # API Settings
    API_TITLE = "Bill Extraction API"
    API_VERSION = __version__
    API_DESCRIPTION = "Extract line item data from medical bills"
    
    # Processing Settings
    MAX_FILE_SIZE_MB = 50
    DEFAULT_DPI = 300
    OCR_CONFIDENCE_THRESHOLD = 0.6
    
    # LLM Settings
    MAX_TOKENS = 4096
    TEMPERATURE = 0.0  # Deterministic for consistency
    
    # Validation Settings
    AMOUNT_TOLERANCE = 0.01
    
    # Performance Settings
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3
    
    # Fraud Detection Settings
    WHITE_PATCH_THRESHOLD = 0.15
    FONT_VARIATION_THRESHOLD = 1000
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }


# Health check data
def get_health_status() -> Dict[str, Any]:
    """
    Get application health status
    
    Returns:
        Health status dictionary
    """
    return {
        "status": "healthy",
        "version": __version__,
        "services": {
            "api": "operational",
            "ocr": "operational",
            "llm": "operational"
        }
    }
