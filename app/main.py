from fastapi import FastAPI, UploadFile, HTTPException
from .utils import DocumentUtils, OCRUtils, TextUtils, ErrorUtils
from .schemas import BillExtractionResponse, BillData, TokenUsage
from .config import Config
import io
from PIL import Image

app = FastAPI()

@app.post("/extract-bill-data", response_model=BillExtractionResponse)
async def extract_bill_data(file: UploadFile):
    try:
        # Load raw bytes
        content = await file.read()

        # Determine type
        doc_type = DocumentUtils.detect_document_type(content)

        # Convert to images
        if doc_type == "pdf":
            pages = DocumentUtils.pdf_to_images(content, dpi=300)
        else:
            pages = [content]

        all_line_items = []
        pagewise_output = []

        for i, img_bytes in enumerate(pages, start=1):
            pil_image = Image.open(io.BytesIO(img_bytes))

            # Preprocess
            processed = OCRUtils.preprocess_image(pil_image)

            # OCR
            text = pytesseract.image_to_string(processed)

            # Extract line items
            items, subtotal = OCRUtils.extract_line_items_from_text(text)

            pagewise_output.append({
                "page_number": i,
                "bill_items": items
            })

            all_line_items.extend(items)

        return BillExtractionResponse(
            is_success=True,
            token_usage=TokenUsage(total_tokens=0, input_tokens=0, output_tokens=0),
            data=BillData(
                pagewise_line_items=pagewise_output,
                total_item_count=len(all_line_items)
            )
        )

    except Exception as e:
        return ErrorUtils.format_error_response(e)
