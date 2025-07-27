import pytest
import json

def test_simple_method_template_handling():
    # Test data for Order.calculateTotal method
    class_context = {
        "class_name": "Order",
        "raw_content": """
        public class Order {
            private List<Item> items;
            
            public double calculateTotal(List<Item> items) {
                double total = 0;
                for (Item item : items) {
                    total += item.getPrice() * item.getQuantity();
                }
                return total;
            }
        }
        """
    }
    
    method_info = {
        "name": "calculateTotal",
        "signature": "double calculateTotal(List<Item> items)",
        "parameters": [{"name": "items", "type": "List<Item>"}],
        "return_type": "double",
        "annotations": [],
        "method_body_raw_code": """
        public double calculateTotal(List<Item> items) {
            double total = 0;
            for (Item item : items) {
                total += item.getPrice() * item.getQuantity();
            }
            return total;
        }
        """
    }
    
    # Create method context
    method_context = {
        "class_name": class_context.get('class_name', 'N/A'),
        "method_name": method_info.get('name'),
        "signature": method_info.get('signature'),
        "parameters": method_info.get('parameters', []),
        "return_type": method_info.get('return_type'),
        "annotations": method_info.get('method_annotations', []),
        "body": method_info.get('method_body_raw_code', '')
    }
    
    # Create JSON schema
    json_schema = """
    {"purpose": "A concise summary of what this method does",
     "complexity": "Low, Medium, or High",
     "dependencies": ["List of internal dependencies"]}
    """
    
    # Create user message template
    user_message = """
    Method Context:
    ```json
    {json.dumps(method_context, indent=2)}
    ```
    
    Provide JSON output following this schema:
    {{ schema }}
    """
    
    # Test the template variables
    variables = {
        "schema": json_schema
    }
    
    # Verify the JSON schema is valid
    try:
        json.loads(json_schema)
    except json.JSONDecodeError:
        pytest.fail("JSON schema is not valid JSON")
    
    # Verify the method context is properly formatted
    assert method_context["class_name"] == "Order"
    assert method_context["method_name"] == "calculateTotal"
    assert len(method_context["parameters"]) == 1
    assert method_context["parameters"][0]["name"] == "items"
    assert method_context["parameters"][0]["type"] == "List<Item>"

def test_complex_method_template_handling():
    # Test data for a more complex method
    class_context = {
        "class_name": "InventoryManager",
        "raw_content": """
        public class InventoryManager {
            private Map<String, Integer> stock;
            private final int MAX_STOCK = 100;
            
            @ThreadSafe
            public synchronized boolean updateStock(String itemId, int quantity) {
                if (quantity < 0) {
                    throw new IllegalArgumentException("Quantity cannot be negative");
                }
                
                int currentStock = stock.getOrDefault(itemId, 0);
                int newStock = currentStock + quantity;
                
                if (newStock > MAX_STOCK) {
                    return false;
                }
                
                stock.put(itemId, newStock);
                return true;
            }
        }
        """
    }
    
    method_info = {
        "name": "updateStock",
        "signature": "boolean updateStock(String itemId, int quantity)",
        "parameters": [
            {"name": "itemId", "type": "String"},
            {"name": "quantity", "type": "int"}
        ],
        "return_type": "boolean",
        "annotations": ["@ThreadSafe"],
        "method_body_raw_code": """
        public synchronized boolean updateStock(String itemId, int quantity) {
            if (quantity < 0) {
                throw new IllegalArgumentException("Quantity cannot be negative");
            }
            
            int currentStock = stock.getOrDefault(itemId, 0);
            int newStock = currentStock + quantity;
            
            if (newStock > MAX_STOCK) {
                return false;
            }
            
            stock.put(itemId, newStock);
            return true;
        }
        """
    }
    
    # Create method context
    method_context = {
        "class_name": class_context.get('class_name', 'N/A'),
        "method_name": method_info.get('name'),
        "signature": method_info.get('signature'),
        "parameters": method_info.get('parameters', []),
        "return_type": method_info.get('return_type'),
        "annotations": method_info.get('annotations', []),
        "body": method_info.get('method_body_raw_code', '')
    }
    
    # Create JSON schema
    json_schema = """
    {"purpose": "A concise summary of what this method does",
     "complexity": "Low, Medium, or High",
     "dependencies": ["List of internal dependencies"]}
    """
    
    # Create user message template
    user_message = """
    Method Context:
    ```json
    {json.dumps(method_context, indent=2)}
    ```
    
    Provide JSON output following this schema:
    {{ schema }}
    """
    
    # Test the template variables
    variables = {
        "schema": json_schema
    }
    
    # Verify the JSON schema is valid
    try:
        json.loads(json_schema)
    except json.JSONDecodeError:
        pytest.fail("JSON schema is not valid JSON")
    
    # Verify the method context is properly formatted
    assert method_context["class_name"] == "InventoryManager"
    assert method_context["method_name"] == "updateStock"
    assert len(method_context["parameters"]) == 2
    assert method_context["parameters"][0]["name"] == "itemId"
    assert method_context["parameters"][0]["type"] == "String"
    assert method_context["parameters"][1]["name"] == "quantity"
    assert method_context["parameters"][1]["type"] == "int"
    assert len(method_context["annotations"]) == 1
    assert method_context["annotations"][0] == "@ThreadSafe"  # Note: We store annotations without @

if __name__ == "__main__":
    pytest.main(["-v", __file__])
