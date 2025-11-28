from typing import List, Dict, Any

class BillValidator:
    
    @staticmethod
    def validate_items(page_data: Dict[str, Any]) -> List[str]:
        """Validate extracted line items"""
        errors = []
        
        for idx, item in enumerate(page_data.get("bill_items", [])):
            # Validate calculation
            expected_amount = item["item_rate"] * item["item_quantity"]
            actual_amount = item["item_amount"]
            
            # Allow 0.01 tolerance for rounding
            if abs(expected_amount - actual_amount) > 0.01:
                errors.append(
                    f"Item {idx+1} '{item['item_name']}': "
                    f"Rate×Qty = {expected_amount} but Amount = {actual_amount}"
                )
        
        return errors
    
    @staticmethod
    def detect_duplicates(all_pages: List[Dict[str, Any]]) -> List[str]:
        """Detect duplicate line items across pages"""
        seen_items = {}
        duplicates = []
        
        for page in all_pages:
            for item in page.get("bill_items", []):
                key = (item["item_name"], item["item_amount"])
                if key in seen_items:
                    duplicates.append(f"Duplicate: {item['item_name']}")
                else:
                    seen_items[key] = True
        
        return duplicates
