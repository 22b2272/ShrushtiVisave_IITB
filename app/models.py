from pydantic import BaseModel, Field
from typing import List, Optional

class BillItem(BaseModel):
    item_name: str
    item_amount: float
    item_rate: float
    item_quantity: float

class PagewiseLineItems(BaseModel):
    page_no: str
    page_type: str = Field(..., description="Bill Detail | Final Bill | Pharmacy")
    bill_items: List[BillItem]

class TokenUsage(BaseModel):
    total_tokens: int
    input_tokens: int
    output_tokens: int

class BillData(BaseModel):
    pagewise_line_items: List[PagewiseLineItems]
    total_item_count: int

class BillExtractionResponse(BaseModel):
    is_success: bool
    token_usage: TokenUsage
    data: BillData

class BillExtractionRequest(BaseModel):
    document: str  # URL to the document
