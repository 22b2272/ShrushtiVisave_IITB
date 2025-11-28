from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

from .models import BillExtractionRequest, BillExtractionResponse, TokenUsage, BillData
from .preprocessing import DocumentPreprocessor
from .ocr_engine import OCREngine
from .extraction import BillExtractor
from .validation import BillValidator
from .utils import (
    DocumentUtils, 
    PerformanceUtils, 
    ErrorUtils,
    ValidationUtils
)
from . import __version__, get_app_info, get_health_status, Config

logger = logging.getLogger(__name__)

app = FastAPI(
    title=Config.API_TITLE,
    version=Config.API_VERSION,
    description=Config.API_DESCRIPTION
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return get_app_info()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return get_health_status()

@app.post("/extract-bill-data", response_model=BillExtractionResponse)
@PerformanceUtils.timer
async def extract_bill_data(request: BillExtractionRequest):
    """
    Extract line item data from medical bills
    """
    try:
        logger.info(f"Processing document: {request.document}")
        
        # Step 1: Download document
        document_bytes = DocumentUtils.download_document(
            request.document, 
            timeout=Config.REQUEST_TIMEOUT
        )
        
        # Validate file size
        file_size_mb = len(document_bytes) / (1024 * 1024)
        if file_size_mb > Config.MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=f"File size ({file_size_mb:.2f}MB) exceeds limit ({Config.MAX_FILE_SIZE_MB}MB)"
            )
        
        # Detect document type
        doc_type = DocumentUtils.detect_document_type(document_bytes)
        logger.info(f"Document type: {doc_type}")
        
        # Convert PDF to images if needed
        if doc_type == 'pdf':
            image_list = DocumentUtils.pdf_to_images(document_bytes, dpi=Config.DEFAULT_DPI)
        else:
            image_list = [document_bytes]
        
        # Process all pages
        all_page_data = []
        extractor = BillExtractor()
        
        for page_num, image_bytes in enumerate(image_list, start=1):
            logger.info(f"Processing page {page_num}/{len(image_list)}")
            
            # Step 2: Preprocess image
            preprocessor = DocumentPreprocessor()
            enhanced_image = preprocessor.enhance_image(image_bytes)
            fraud_flags = preprocessor.detect_fraud(image_bytes)
            
            if fraud_flags:
                logger.warning(f"Page {page_num} fraud indicators: {fraud_flags}")
            
            # Step 3: OCR extraction
            ocr_engine = OCREngine()
            ocr_text = ocr_engine.extract_text(enhanced_image)
            
            # Step 4: LLM extraction
            page_data = extractor.extract_from_text(ocr_text, page_number=page_num)
            
            # Step 5: Validation
            validator = BillValidator()
            validation_errors = validator.validate_items(page_data)
            
            if validation_errors:
                logger.warning(f"Page {page_num} validation issues: {validation_errors}")
            
            all_page_data.append(page_data)
        
        # Check for duplicates across pages
        duplicate_items = BillValidator.detect_duplicates(all_page_data)
        if duplicate_items:
            logger.warning(f"Duplicate items detected: {duplicate_items}")
        
        # Calculate total items
        total_items = sum(len(page["bill_items"]) for page in all_page_data)
        
        # Calculate total amount for logging
        total_amount = ValidationUtils.calculate_total(
            [item for page in all_page_data for item in page["bill_items"]]
        )
        logger.info(f"Total extracted amount: ₹{total_amount:.2f}")
        
        # Build response
        response_data = BillExtractionResponse(
            is_success=True,
            token_usage=TokenUsage(**extractor.get_token_usage()),
            data=BillData(
                pagewise_line_items=all_page_data,
                total_item_count=total_items
            )
        )
        
        logger.info(f"Successfully extracted {total_items} items from {len(image_list)} pages")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        error_response = ErrorUtils.format_error_response(e)
        raise HTTPException(status_code=500, detail=error_response)

@app.get("/config")
async def get_config():
    """Get current configuration"""
    return Config.to_dict()
