import anthropic
import os
import json
from typing import Dict, Any, Tuple

class BillExtractor:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.total_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0
    
    def extract_from_text(self, ocr_text: str, page_number: int) -> Dict[str, Any]:
        """Extract structured data using Claude"""
        
        prompt = f"""You are extracting line items from a medical bill. Extract ALL line items with exact details.

OCR Text from Page {page_number}:
{ocr_text}

CRITICAL RULES:
1. Extract EVERY line item (do not miss any)
2. Item names EXACTLY as written in the bill
3. Calculate: item_amount = item_rate × item_quantity
4. Do NOT include sub-totals or final totals as line items
5. Classify page_type as: "Bill Detail", "Final Bill", or "Pharmacy"

Return ONLY valid JSON (no markdown, no explanations):
{{
  "page_no": "{page_number}",
  "page_type": "Bill Detail",
  "bill_items": [
    {{
      "item_name": "exact name from bill",
      "item_amount": 0.00,
      "item_rate": 0.00,
      "item_quantity": 0.00
    }}
  ]
}}"""

        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Track token usage
        self.input_tokens += message.usage.input_tokens
        self.output_tokens += message.usage.output_tokens
        self.total_tokens += message.usage.input_tokens + message.usage.output_tokens
        
        # Extract JSON from response
        response_text = message.content[0].text
        
        # Remove markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        return json.loads(response_text.strip())
    
    def get_token_usage(self) -> Dict[str, int]:
        """Return cumulative token usage"""
        return {
            "total_tokens": self.total_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens
        }
