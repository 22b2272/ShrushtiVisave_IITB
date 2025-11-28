from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
from typing import List
import logging

from .models import BillExtractionRequest, BillExtractionResponse, TokenUsage, BillData
from .preprocessing import DocumentPreprocessor
from .ocr_engine import OCREngine
from .extraction import BillExtractor
from .validation import BillValidator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Bill Extraction API", version="1.0.0")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/extract-bill-data", response_model=BillExtractionResponse)
async def extract_bill_data(request: BillExtractionRequest):
    """
    Extract line item data from medical bills
    """
    try:
        logger.info(f"Processing document: {request.document}")
        
        # Step 1: Download document
        response = requests.get(request.document, timeout=30)
        response.raise_for_status()
        document_bytes = response.content
        
        # Step 2: Preprocess image
        preprocessor = DocumentPreprocessor()
        enhanced_image = preprocessor.enhance_image(document_bytes)
        fraud_flags = preprocessor.detect_fraud(document_bytes)
        
        if fraud_flags:
            logger.warning(f"Fraud indicators: {fraud_flags}")
        
        # Step 3: OCR extraction
        ocr_engine = OCREngine()
        ocr_text = ocr_engine.extract_text(enhanced_image)
        
        # Step 4: LLM extraction
        extractor = BillExtractor()
        page_data = extractor.extract_from_text(ocr_text, page_number=1)
        
        # Step 5: Validation
        validator = BillValidator()
        validation_errors = validator.validate_items(page_data)
        
        if validation_errors:
            logger.warning(f"Validation issues: {validation_errors}")
        
        # Step 6: Build response
        total_items = len(page_data["bill_items"])
        
        response_data = BillExtractionResponse(
            is_success=True,
            token_usage=TokenUsage(**extractor.get_token_usage()),
            data=BillData(
                pagewise_line_items=[page_data],
                total_item_count=total_items
            )
        )
        
        logger.info(f"Successfully extracted {total_items} items")
        return response_data
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
