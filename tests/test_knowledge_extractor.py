import pytest
import json
from unittest.mock import Mock, patch
from knowledge_extractor import KnowledgeExtractor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

# Mock the LLM
mock_llm = Mock()
mock_llm.invoke.return_value = {
    "purpose": "Calculates the total price of items in an order",
    "complexity": "Low",
    "dependencies": ["Item.getPrice", "Item.getQuantity"]
}

def test_prompt_template_handling():
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
        "annotations": method_info.get('annotations', []),
        "body": method_info.get('method_body_raw_code', '')
    }
    
    # Create JSON schema with proper escaping
    schema_dict = {
        "purpose": "A concise summary of what this method does",
        "complexity": "Low, Medium, or High",
        "dependencies": ["List of internal dependencies"]
    }
    json_schema = json.dumps(schema_dict)
    
    # Create user message with proper LangChain template syntax
    user_message = """
    Method Context:
    ```json
    {{ method_context }}
    ```
    
    Provide JSON output following this schema:
    {{ schema }}
    """
    
    # Test the template variables
    variables = {
        "schema": json_schema,
        "method_context": json.dumps(method_context, indent=2)
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
    
    # Test prompt template formatting
    try:
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert Java developer analyzing method implementations."),
            ("user", user_message)
        ])
        
        # Create a mock LLM
        mock_llm = Mock()
        mock_llm.invoke.return_value = "Test response"
        
        # Create a chain and invoke it using the same pattern as KnowledgeExtractor._extract_method_data
        chain = prompt | mock_llm
        
        # Mock the LLM's invoke method to return a valid JSON response
        mock_llm.invoke.return_value = {
            "purpose": "A concise summary of what this method does",
            "complexity": "Low",
            "dependencies": ["List of internal dependencies"]
        }
        
        # First verify the template substitution works
        formatted_message = prompt.format_messages(**variables)
        user_content = formatted_message[1].content
        assert "{{schema}}" not in user_content
        assert "{{method_context}}" not in user_content
        assert "Method Context:" in user_content
        assert "```json" in user_content
        assert "```" in user_content
        assert "Provide JSON output following this schema:" in user_content
        
        # Verify the JSON schema structure
        assert "purpose" in user_content
        assert "complexity" in user_content
        assert "dependencies" in user_content
        assert "List of internal dependencies" in user_content
        assert "Low, Medium, or High" in user_content
        assert "A concise summary of what this method does" in user_content
        
        # Now verify the chain invocation
        response = chain.invoke(variables)
        
        # Verify the prompt template was formatted correctly
        assert len(formatted_messages) == 2  # Should be system and user messages
        user_message = next(msg for msg in formatted_messages if msg.type == "human").content
        assert "{{schema}}" not in user_message
        assert "{{method_context}}" not in user_message
        assert "Method Context:" in user_message
        assert "```json" in user_message
        assert "```" in user_message
        assert "Provide JSON output following this schema:" in user_message
        assert "purpose" in user_message
        assert "complexity" in user_message
        assert "dependencies" in user_message
        assert "List of internal dependencies" in user_message
        assert "Low, Medium, or High" in user_message
        assert "A concise summary of what this method does" in user_message
        assert method_context["class_name"] in user_message
        assert method_context["method_name"] in user_message
        assert method_context["signature"] in user_message
        
        # Now verify the chain invocation
        response = chain.invoke(variables)
        mock_llm.invoke.assert_called_once()
        args = mock_llm.invoke.call_args[0][0]
        assert isinstance(args, list)  # Should be a list of messages
        assert len(args) == 2  # Should be system and user messages
    except Exception as e:
        pytest.fail(f"Prompt template formatting failed: {str(e)}")

def test_complex_method_handling():
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
    
    # Create JSON schema with proper escaping
    schema_dict = {
        "purpose": "A concise summary of what this method does",
        "complexity": "Low, Medium, or High",
        "dependencies": ["List of internal dependencies"]
    }
    json_schema = json.dumps(schema_dict)
    
    # Create user message with proper LangChain template syntax
    user_message = """
    Method Context:
    ```json
    {{ method_context }}
    ```
    
    Provide JSON output following this schema:
    {{ schema }}
    """
    
    # Test the template variables
    variables = {
        "schema": json_schema,
        "method_context": json.dumps(method_context, indent=2)
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
    
    # Test prompt template formatting
    try:
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert Java developer analyzing method implementations."),
            ("user", user_message)
        ])
        
        # Create a mock LLM
        mock_llm = Mock()
        mock_llm.invoke.return_value = "Test response"
        
        # Create a chain and invoke it using the same pattern as KnowledgeExtractor._extract_method_data
        chain = prompt | mock_llm
        
        # Mock the LLM's invoke method to return a valid JSON response
        mock_llm.invoke.return_value = {
            "purpose": "A concise summary of what this method does",
            "complexity": "Low",
            "dependencies": ["List of internal dependencies"]
        }
        
        # First verify the template substitution works
        formatted_message = prompt.format_messages(**variables)
        user_content = formatted_message[1].content
        assert "{{schema}}" not in user_content
        assert "{{method_context}}" not in user_content
        assert "Method Context:" in user_content
        assert "```json" in user_content
        assert "```" in user_content
        assert "Provide JSON output following this schema:" in user_content
        
        # Verify the JSON schema structure
        assert "purpose" in user_content
        assert "complexity" in user_content
        assert "dependencies" in user_content
        assert "List of internal dependencies" in user_content
        assert "Low, Medium, or High" in user_content
        assert "A concise summary of what this method does" in user_content
        
        # Now verify the chain invocation
        response = chain.invoke(variables)
        
        # Verify the prompt template was formatted correctly
        assert len(formatted_messages) == 2  # Should be system and user messages
        user_message = next(msg for msg in formatted_messages if msg.type == "human").content
        assert "{{schema}}" not in user_message
        assert "{{method_context}}" not in user_message
        assert "Method Context:" in user_message
        assert "```json" in user_message
        assert "```" in user_message
        assert "Provide JSON output following this schema:" in user_message
        assert "purpose" in user_message
        assert "complexity" in user_message
        assert "dependencies" in user_message
        assert "List of internal dependencies" in user_message
        assert "Low, Medium, or High" in user_message
        assert "A concise summary of what this method does" in user_message
        assert method_context["class_name"] in user_message
        assert method_context["method_name"] in user_message
        assert method_context["signature"] in user_message
        
        # Now verify the chain invocation
        response = chain.invoke(variables)
        mock_llm.invoke.assert_called_once()
        args = mock_llm.invoke.call_args[0][0]
        assert isinstance(args, list)  # Should be a list of messages
        assert len(args) == 2  # Should be system and user messages
    except Exception as e:
        pytest.fail(f"Prompt template formatting failed: {str(e)}")
    assert method_context["parameters"][0]["name"] == "itemId"
    assert method_context["parameters"][0]["type"] == "String"
    assert method_context["parameters"][1]["name"] == "quantity"
    assert method_context["parameters"][1]["type"] == "int"
    assert method_context["return_type"] == "boolean"
    assert method_context["annotations"] == ["@ThreadSafe"]  # Note: We store annotations without @
    # Create a test method context
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
    
    # Create a test extractor with mocked LLM
    extractor = KnowledgeExtractor()
    extractor.llm = mock_llm
    
    # Test the prompt template creation
    parser = JsonOutputParser()
    json_schema = """
    {"purpose": "A concise summary of what this method does",
     "complexity": "Low, Medium, or High",
     "dependencies": ["List of internal dependencies"]}
    """
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        Analyze Java method and provide structured output with these fields:
        {{ schema }}
        """),
        ("user", """
        Method Context:
        ```json
        {json.dumps(method_context, indent=2)}
        ```
        
        Provide JSON output following the schema above.
        """)
    ])
    
    # Create variables dictionary
    variables = {
        "schema": json_schema
    }
    
    # Test the LLM invocation
    response = extractor._invoke_llm(prompt, parser, variables)
    
    # Verify the response
    assert response is not None
    assert isinstance(response, dict)
    assert "purpose" in response
    assert "complexity" in response
    assert "dependencies" in response

if __name__ == "__main__":
    pytest.main(["-v", __file__])
