from pydantic import BaseModel
from typing import List, Dict

class TokenUsage(BaseModel):
    total_tokens: int
    input_tokens: int
    output_tokens: int

class BillData(BaseModel):
    pagewise_line_items: List[Dict]
    total_item_count: int

class BillExtractionResponse(BaseModel):
    is_success: bool
    token_usage: TokenUsage
    data: BillData
