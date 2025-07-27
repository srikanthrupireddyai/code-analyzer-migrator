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

def test_class_description_extraction():
    # Test data for a sample class
    class_context = {
        "class_name": "CustomerService",
        "package": "com.example.service",
        "doc": """/**
         * Service class for managing customer operations.
         * Handles customer creation, retrieval, and updates.
         */""",
        "signature": "public class CustomerService implements CustomerServiceInterface",
        "body": """
        private CustomerRepository customerRepo;
        
        public CustomerService(CustomerRepository repo) {
            this.customerRepo = repo;
        }
        
        public Customer getCustomerById(int id) {
            return customerRepo.findById(id);
        }
        """
    }
    
    # Create mock LLM response
    mock_llm_response = Mock()
    mock_llm_response.content = json.dumps({
        "purpose": "Service class for managing customer operations. Handles customer creation, retrieval, and updates.",
        "responsibilities": ["Manage customer operations", "Handle customer creation", "Handle customer retrieval", "Handle customer updates"],
        "relationships": ["CustomerRepository", "CustomerServiceInterface"]
    })
    
    # Create mock LLM
    mock_llm = Mock()
    mock_llm.invoke.return_value = mock_llm_response
    
    # Create KnowledgeExtractor instance with mock LLM
    extractor = KnowledgeExtractor()
    extractor.llm = mock_llm
    
    # Test class description extraction
    result = extractor._extract_class_description(class_context)
    
    # Verify LLM was called with correct prompt
    mock_llm.invoke.assert_called()
    
    # Verify the prompt was properly formatted
    call_args = mock_llm.invoke.call_args[1].get('variables', {})
    assert "class_name" in call_args
    assert call_args["class_name"] == "CustomerService"
    assert "package" in call_args
    assert "doc" in call_args
    assert "signature" in call_args
    assert "body" in call_args
    
    # Verify the prompt was properly formatted
    call_args = mock_chain.invoke.call_args[1].get('variables', {})
    assert "class_name" in call_args
    assert call_args["class_name"] == "CustomerService"
    assert "package" in call_args
    assert "doc" in call_args
    assert "signature" in call_args
    assert "body" in call_args
    
    # Verify result structure
    assert isinstance(result, dict)
    assert "class_name" in result
    assert result["class_name"] == "CustomerService"
    assert "package" in result
    assert result["package"] == "com.example.service"
    assert "purpose" in result
    assert isinstance(result["purpose"], str)
    assert len(result["purpose"]) > 0
    assert "responsibilities" in result
    assert isinstance(result["responsibilities"], list)
    assert len(result["responsibilities"]) > 0
    assert "relationships" in result
    assert isinstance(result["relationships"], list)
    assert len(result["relationships"]) > 0
    
    # Verify LLM was called with correct prompt
    extractor.llm.invoke.assert_called()
    
    # Verify the prompt was properly formatted
    call_args = extractor.llm.invoke.call_args[1].get('variables', {})
    assert "class_name" in call_args
    assert "package" in call_args
    assert "doc" in call_args
    assert "signature" in call_args
    assert "body" in call_args

def test_method_extraction():
    # Test data for a sample method
    method_context = {
        "class_name": "CustomerService",
        "package": "com.example.service",
        "method_name": "getCustomerById",
        "signature": "Customer getCustomerById(int id)",
        "parameters": [{"name": "id", "type": "int"}],
        "return_type": "Customer",
        "annotations": [],
        "body": """
        public Customer getCustomerById(int id) {
            return customerRepo.findById(id);
        }
        """
    }
    
    # Create KnowledgeExtractor instance with mock LLM
    extractor = KnowledgeExtractor()
    extractor.llm = Mock()
    extractor.llm.invoke.return_value = {
        "content": json.dumps({"name": "getCustomerById",
                              "signature": "Customer getCustomerById(int id)",
                              "description": "Retrieves customer data by ID",
                              "complexity": "Low"})
    }
    
    # Test method extraction
    result = extractor._extract_method_data(method_context, method_context)
    
    # Verify LLM was called with correct prompt
    extractor.llm.invoke.assert_called()
    
    # Verify result structure
    assert isinstance(result, dict)
    assert "name" in result
    assert "signature" in result
    assert "description" in result
    assert "complexity" in result
    assert result["name"] == "getCustomerById"
    assert result["complexity"] in ["Low", "Medium", "High"]
    
    # Verify LLM was called with correct prompt
    extractor.llm.invoke.assert_called()
    
    # Verify the prompt was properly formatted
    call_args = extractor.llm.invoke.call_args[1].get('variables', {})
    assert "class_name" in call_args
    assert "package" in call_args
    assert "method_name" in call_args
    assert "signature" in call_args
    assert "parameters" in call_args
    assert "return_type" in call_args
    assert "annotations" in call_args
    assert "body" in call_args

def test_extract_all_knowledge():
    # Test data for multiple classes
    parsed_java_data = [
        {
            "class_name": "CustomerService",
            "package": "com.example.service",
            "doc": "Service class for managing customer operations.",
            "signature": "public class CustomerService implements CustomerServiceInterface",
            "methods": [
                {
                    "name": "getCustomerById",
                    "signature": "Customer getCustomerById(int id)",
                    "parameters": [{"name": "id", "type": "int"}],
                    "return_type": "Customer",
                    "annotations": [],
                    "method_body_raw_code": """
                    public Customer getCustomerById(int id) {
                        return customerRepo.findById(id);
                    }
                    """
                }
            ]
        },
        {
            "class_name": "CustomerRepository",
            "package": "com.example.repository",
            "doc": "Repository class for customer data access.",
            "signature": "public class CustomerRepository",
            "methods": [
                {
                    "name": "findById",
                    "signature": "Customer findById(int id)",
                    "parameters": [{"name": "id", "type": "int"}],
                    "return_type": "Customer",
                    "annotations": [],
                    "method_body_raw_code": """
                    public Customer findById(int id) {
                        return customers.get(id);
                    }
                    """
                }
            ]
        }
    ]
    
    # Create mock LLM that returns valid responses
    mock_llm = Mock()
    mock_llm.invoke.return_value = {
        "content": json.dumps({"description": "Service class for managing customer operations. Handles customer creation, retrieval, and updates."})
    }
    
    # Create KnowledgeExtractor instance with mock LLM
    extractor = KnowledgeExtractor()
    extractor.llm = mock_llm
    
    # Test extract_all_knowledge
    result = extractor.extract_all_knowledge(parsed_java_data, None)
    
    # Verify LLM was called multiple times
    assert mock_llm.invoke.call_count > 0
    
    # Verify result structure
    assert isinstance(result, dict)
    assert "projectOverview" in result
    assert "modules" in result
    assert isinstance(result["modules"], list)
    
    # Verify each module has correct structure
    for module in result["modules"]:
        assert "name" in module
        assert isinstance(module["name"], str)
        assert "description" in module
        assert isinstance(module["description"], str)
        assert "methods" in module
        assert isinstance(module["methods"], list)
        
        # Verify each method has correct structure
        for method in module["methods"]:
            assert "name" in method
            assert isinstance(method["name"], str)
            assert "signature" in method
            assert isinstance(method["signature"], str)
            assert "description" in method
            assert isinstance(method["description"], str)
            assert "complexity" in method
            assert isinstance(method["complexity"], str)
            assert method["complexity"] in ["Low", "Medium", "High"]
    
    # Verify LLM was called multiple times
    assert mock_llm.invoke.call_count > 0
    
    # Verify the number of LLM calls matches the number of classes and methods
    expected_calls = len(parsed_java_data) + sum(len(cls["methods"]) for cls in parsed_java_data)
    assert mock_llm.invoke.call_count == expected_calls

def test_complex_method_handling():
    # Test data for a more complex method with annotations and complex logic
    class_context = {
        "class_name": "InventoryManager",
        "package": "com.example.inventory",
        "doc": """/**
         * Manages inventory levels with thread safety.
         */""",
        "signature": "public class InventoryManager",
        "body": """
        private final Map<String, Integer> stock = new ConcurrentHashMap<>();
        private static final int MAX_STOCK = 1000;
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
    
    # Create KnowledgeExtractor instance with mock LLM
    extractor = KnowledgeExtractor()
    extractor.llm = Mock()
    extractor.llm.invoke.return_value = {
        "content": json.dumps({"name": "updateStock",
                              "signature": "boolean updateStock(String itemId, int quantity)",
                              "description": "Updates inventory stock levels with thread safety",
                              "complexity": "Medium"})
    }
    
    # Test method extraction
    result = extractor._extract_method_data(method_context, class_context)
    
    # Verify LLM was called with correct prompt
    extractor.llm.invoke.assert_called()
    
    # Verify result structure
    assert isinstance(result, dict)
    assert "name" in result
    assert "signature" in result
    assert "description" in result
    assert "complexity" in result
    assert result["name"] == "updateStock"
    assert result["complexity"] in ["Low", "Medium", "High"]
    
    # Verify LLM was called with correct prompt
    extractor.llm.invoke.assert_called()
    
    # Verify the prompt was properly formatted
    call_args = extractor.llm.invoke.call_args[1].get('variables', {})
    assert "class_name" in call_args
    assert "package" in call_args
    assert "method_name" in call_args
    assert "signature" in call_args
    assert "parameters" in call_args
    assert "return_type" in call_args
    assert "annotations" in call_args
    assert "body" in call_args
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
