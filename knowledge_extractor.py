import json
from pathlib import Path
import os
import logging
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime, timedelta
import threading
import hashlib
import re
import pickle

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Semaphore


# Error messages
ERROR_MESSAGES = {
    "missing_input": "Method info and class context are required",
    "missing_method_name": "Method name is missing",
    "missing_fields": "Missing required fields in method info: {}",
    "invalid_schema": "Invalid JSON schema: {}",
    "invalid_dict": "JSON schema must be a dictionary",
    "invalid_params": "Invalid parameter format: {}",
    "invalid_response": "LLM response must be a dictionary",
    "missing_response_fields": "Missing required response fields: {}"
}

# File paths
PARSED_JAVA_DATA_FILE = "parsed_java_data.json"

# Helper functions

def validate_method_info(method_info: dict) -> None:
    """Validate the structure of method information."""
    required_fields = ['name', 'signature', 'body', 'parameters']
    missing_fields = [f for f in required_fields if f not in method_info]
    if missing_fields:
        raise ValueError(ERROR_MESSAGES['missing_fields'].format(missing_fields))

def create_method_context(class_context: dict, method_info: dict) -> dict:
    """Create the context needed for method analysis."""
    return {
        'class_name': class_context.get('class_name', ''),
        'signature': method_info['signature'],
        'body': method_info['body'],
        'parameters': method_info.get('parameters', []),
        'imports': class_context.get('imports', []),
        'class_doc': class_context.get('doc', ''),
        'method_doc': method_info.get('doc', '')
    }

def create_prompt(schema: dict, context: dict, attempt: int = 1) -> ChatPromptTemplate:
    """Create a prompt for the LLM based on the schema and context."""
    base_instructions = """
    You are an expert Java developer analyzing code. Your task is to analyze the following method and provide detailed information about it.
    
    Guidelines:
    1. Focus on the actual functionality of the method, not just its name
    2. Consider the method's role within its class and the overall system
    3. Look for patterns in the code that indicate common design patterns or architectural decisions
    4. Be specific about dependencies and usage patterns
    """
    
    # Add retry-specific instructions for subsequent attempts
    if attempt > 1:
        base_instructions += f"\n\nThis is attempt {attempt}. Please refine your previous analysis based on the feedback received."
    
    return ChatPromptTemplate.from_messages([
        ("system", "You are an expert Java developer analyzing code."),
        ("user", f"""
        {base_instructions}

        Method Context:
        Class: {context['class_name']}
        Signature: {context['signature']}
        Parameters: {json.dumps(context['parameters'], indent=2)}
        Imports: {', '.join(context['imports'])}
        Class Documentation: {context['class_doc']}
        Method Documentation: {context['method_doc']}

        Method Body:
        {context['body']}

        Please analyze this method and provide the following information in JSON format:
        {json.dumps(schema, indent=2)}
        """)
    ])

def validate_response(response_data: dict) -> None:
    """Validate the structure of the LLM response data."""
    required_fields = ['purpose', 'complexity', 'dependencies', 'usage', 'exceptions']
    missing_fields = [f for f in required_fields if f not in response_data]
    if missing_fields:
        raise ValueError(f"Missing required fields in response: {missing_fields}")
    
    # Validate complexity level
    if response_data['complexity'] not in [level.value for level in ComplexityLevel]:
        raise ValueError(f"Invalid complexity level: {response_data['complexity']}")

def validate_parameters(parameters: list[dict]) -> None:
    """Validate the structure of method parameters."""
    if not isinstance(parameters, list):
        raise ValueError("Parameters must be a list")
    
    for param in parameters:
        if not isinstance(param, dict):
            raise ValueError("Each parameter must be a dictionary")
            
        required_fields = ['name', 'type']
        missing_fields = [f for f in required_fields if f not in param]
        if missing_fields:
            raise ValueError(f"Parameter is missing required fields: {missing_fields}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Method complexity levels
class ComplexityLevel(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

# JSON schema for method analysis
METHOD_SCHEMA = {
    "purpose": "A concise summary of what this method does",
    "complexity": "Low, Medium, or High",
    "dependencies": ["List of internal dependencies"],
    "usage": "Example usage of the method",
    "exceptions": ["List of possible exceptions"]
}

class LLMSyncManager:
    """Manages synchronization for LLM calls across threads."""
    
    def __init__(self, max_concurrent: int = 5):
        """
        Initialize the LLM sync manager.
        
        Args:
            max_concurrent: Maximum number of concurrent LLM calls
        """
        self.max_concurrent = max_concurrent
        self._lock = Lock()
        self._semaphore = threading.Semaphore(max_concurrent)
        self._active_calls = 0
        
    def acquire(self):
        """Acquire a slot for an LLM call."""
        self._semaphore.acquire()
        with self._lock:
            self._active_calls += 1
            logger.debug(f"Active LLM calls: {self._active_calls}")
    
    def release(self):
        """Release a slot after LLM call completion."""
        with self._lock:
            self._active_calls -= 1
            logger.debug(f"Active LLM calls: {self._active_calls}")
        self._semaphore.release()
    
    def wait_for_available_slot(self):
        """Wait until a slot becomes available."""
        self._semaphore.acquire()
        self._semaphore.release()


class CacheManager:
    """Manages caching of LLM responses to avoid redundant calls."""
    
    def __init__(self, cache_dir: str = "cache", max_age: timedelta = timedelta(hours=24), max_size: int = 1000):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory to store cache files
            max_age: Maximum age of cache entries
            max_size: Maximum number of cache entries
        """
        self.cache_dir = os.path.abspath(cache_dir)
        self.max_age = max_age
        self.max_size = max_size
        self._cache = {}  # In-memory cache for faster access
        self._lock = threading.Lock()
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clean up expired entries."""
        self._cleanup_expired()
        
    def _get_cache_key(self, method_name: str, method_body: str) -> str:
        """Generate a unique cache key based on method name and body."""
        key = f"{method_name}_{hashlib.sha256(method_body.encode()).hexdigest()[:16]}"
        return os.path.join(self.cache_dir, f"{key}.pkl")
    
    def _cleanup_expired(self):
        """Clean up expired cache entries."""
        with self._lock:
            current_time = datetime.now()
            for key, entry in list(self._cache.items()):
                if current_time - entry['timestamp'] > self.max_age:
                    del self._cache[key]
                    cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
                    if os.path.exists(cache_file):
                        os.remove(cache_file)
            
            # If cache size exceeds limit, remove oldest entries
            if len(self._cache) > self.max_size:
                sorted_entries = sorted(
                    self._cache.items(),
                    key=lambda x: x[1]['timestamp']
                )
                for key, _ in sorted_entries[:len(self._cache) - self.max_size]:
                    del self._cache[key]
                    cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
                    if os.path.exists(cache_file):
                        os.remove(cache_file)
    
    def get(self, method_name: str, method_body: str) -> Optional[dict]:
        """Retrieve cached response if available and not expired."""
        key = self._get_cache_key(method_name, method_body)
        
        # Check in-memory cache first
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if datetime.now() - entry['timestamp'] <= self.max_age:
                    return entry['response']
                else:
                    del self._cache[key]
                    return None
        
        # Check disk cache if not in memory
        try:
            with open(key, 'rb') as f:
                cached_data = pickle.load(f)
                
            # Validate cache entry
            if datetime.now() - cached_data['timestamp'] <= self.max_age:
                with self._lock:
                    self._cache[key] = cached_data
                return cached_data['response']
                
        except Exception as e:
            logger.error(f"Cache retrieval failed: {e}")
            # Try to clean up corrupted cache file
            if os.path.exists(key):
                os.remove(key)
            
        return None
    
    def save(self, method_name: str, method_body: str, response: dict) -> None:
        """Save response to cache."""
        key = self._get_cache_key(method_name, method_body)
        
        try:
            # Save to in-memory cache first
            with self._lock:
                self._cache[key] = {
                    'timestamp': datetime.now(),
                    'response': response
                }
                
                # Clean up if cache exceeds size limit
                if len(self._cache) > self.max_size:
                    self._cleanup_expired()
            
            # Save to disk
            with open(key, 'wb') as f:
                pickle.dump({
                    'timestamp': datetime.now(),
                    'response': response
                }, f)
                
        except Exception as e:
            logger.error(f"Cache save failed: {e}")
            # Remove partial file if save failed
            if os.path.exists(key):
                os.remove(key)

class KnowledgeExtractor:
    def __init__(self, 
                 llm_model_name: str = "gpt-4", 
                 temperature: float = 0.0, 
                 cache_enabled: bool = True, 
                 batch_size: int = 5,
                 max_workers: int = None,
                 max_retries: int = 3,
                 timeout: float = 30.0,
                 max_concurrent_llm: int = 5):
        """
        Initialize the KnowledgeExtractor with LLM configuration.
        
        Args:
            llm_model_name: Name of the LLM model to use
            temperature: Temperature for LLM generation (0.0 for more deterministic output)
            cache_enabled: Whether to enable response caching
            batch_size: Number of methods to process in parallel
            max_workers: Maximum number of worker threads
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
            max_concurrent_llm: Maximum number of concurrent LLM calls
        """
        # Validate API key
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(ERROR_MESSAGES['missing_input'])
            
        self.llm_model_name = llm_model_name
        self.temperature = temperature
        self.api_key = api_key
        self.cache_enabled = cache_enabled
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_concurrent_llm = max_concurrent_llm
        
        # Set max_workers based on system resources if not provided
        if max_workers is None:
            max_workers = min(32, (os.cpu_count() or 1) + 4)  # Default from ThreadPoolExecutor
        self.max_workers = max_workers
        
        try:
            # Initialize LLM with explicit configuration
            self.llm = ChatOpenAI(
                model=self.llm_model_name, 
                temperature=self.temperature,
                openai_api_key=api_key,
                verbose=True,
                timeout=timeout
            )
            logger.info(f"Initialized LLM: {self.llm_model_name}")
            
            # Initialize cache if enabled
            if self.cache_enabled:
                self.cache = CacheManager(max_size=10000)  # Larger cache size
                logger.info("Cache system initialized")
            
            # Initialize LLM sync manager
            self.llm_sync = LLMSyncManager(max_concurrent=self.max_concurrent_llm)
            
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            self.llm = None

    def _invoke_llm(self, prompt: ChatPromptTemplate, parser=None, variables=None, timeout: float = None):
        """Invoke the LLM with the given prompt and optional parser."""
        if not self.llm:
            logger.error("LLM not initialized. Cannot make API call.")
            return None
        try:
            self.llm_sync.acquire()
            try:
                chain = prompt | self.llm
                if parser:
                    chain = chain | parser
                response = chain.invoke(variables or {}, timeout=timeout)
                return response
            finally:
                self.llm_sync.release()
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            return None

    def _extract_class_description(self, class_context: dict) -> Optional[dict]:
        """
        Extract a description of a class using the LLM.
        
        Args:
            class_context: Dictionary containing class context including name, doc, signature, and body
            
        Returns:
            Dictionary containing extracted class description or None if extraction failed
        """
        try:
            # Basic input validation
            if not class_context:
                raise ValueError(ERROR_MESSAGES['missing_input'])
                
            class_name = class_context.get('class_name')
            if not class_name:
                raise ValueError(ERROR_MESSAGES['missing_class_name'])
                
            # Prepare class context data for the prompt
            class_context_data = {
                "class_name": class_name,
                "package": class_context.get('package_name', ''),  # Changed from 'N/A' to empty string
                "doc": class_context.get('doc', 'No documentation available'),
                "signature": f"@{class_context.get('class_annotations', [])}\n" +
                            f"public class {class_name} {class_context.get('extends', '')} {class_context.get('implements', '')} {{",
                "body": "\n".join([
                    f"    {method.get('signature', '')} {{ ... }}" for method in class_context.get('methods', [])
                ])
            }
            
            logger.info(f"Class context data: {class_context_data}")

            # Create the prompt template with proper LangChain message format
            prompt = ChatPromptTemplate.from_messages([
                {"role": "system", "content": "You are an expert software architect analyzing Java code."},
                {"role": "user", "content": """Analyze the following Java class code to provide a concise description of its purpose and functionality.
                
                Class Information:
                Name: {class_name}
                Package: {package}
                
                Documentation:
                {doc}
                
                Code:
                {signature}
                {body}
                
                Please provide a description of the class's purpose and functionality in a concise paragraph.
                Format the response as a JSON object with this structure:
                {{
                    "description": "A concise paragraph describing the class's purpose and functionality"
                }}
                """}
            ])

            # Prepare the variables for the template
            variables = class_context_data

            logger.info(f"Class description prompt variables: {variables}")

            # Create and invoke the chain
            chain = prompt | self.llm
            response = chain.invoke(variables)
            
            if response:
                # Extract the content from the response
                if hasattr(response, 'content'):
                    content = response.content
                    # Clean up the content by removing any extra quotes or formatting
                    content = content.strip().strip('"').strip()
                    try:
                        # Parse the JSON response
                        description_data = json.loads(content)
                        return {
                            "class_name": class_name,
                            "package": class_context.get('package', ''),
                            "purpose": description_data.get('purpose', ''),
                            "responsibilities": description_data.get('responsibilities', []),
                            "relationships": description_data.get('relationships', [])
                        }
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON response for class {class_name}: {e}")
                        return None
                else:
                    logger.error(f"Unexpected response type for class {class_name}: {type(response)}")
                    return None
            else:
                logger.error(f"Failed to get response for class {class_name}")
                return None
        except Exception as e:
            logger.error(f"Failed to extract description for class {class_context.get('class_name', 'unknown')}: {e}")
            return None

    def _extract_method_data(self, method_context: dict) -> Optional[dict]:
        """
        Extract a description of a method using the LLM.
        
        Args:
            method_context: Dictionary containing method context including name, signature, and parameters
            
        Returns:
            Dictionary containing extracted method description or None if extraction failed
        """
        try:
            # Basic input validation
            if not method_context:
                raise ValueError(ERROR_MESSAGES['missing_input'])
                
            method_name = method_context.get('method_name')
            if not method_name:
                raise ValueError(ERROR_MESSAGES['missing_method_name'])
                
            # Prepare method context data for the prompt
            method_context_data = {
                "method_name": method_name,
                "package": method_context.get('package', ''),
                "signature": method_context.get('signature', ''),
                "parameters": method_context.get('parameters', []),
                "annotations": method_context.get('annotations', []),
                "class_name": method_context.get('class_name', '')
            }
            
            logger.info(f"Method context data: {method_context_data}")

            # Create the prompt template with proper LangChain message format
            prompt = ChatPromptTemplate.from_messages([
                {"role": "system", "content": "You are an expert software architect analyzing Java code."},
                {"role": "user", "content": """Analyze the following Java method to provide a concise description of its purpose and functionality.
                
                Method Information:
                Name: {method_name}
                Package: {package}
                Class: {class_name}
                
                Signature:
                {signature}
                
                Parameters:
                {parameters}
                
                Annotations:
                {annotations}
                
                Please provide a description of the method's purpose and functionality in a concise paragraph.
                Format the response as a JSON object with this structure:
                {{
                    "purpose": "A concise paragraph describing the method's purpose and functionality",
                    "parameters": [
                        {{
                            "name": "parameter_name",
                            "type": "parameter_type",
                            "description": "parameter_description"
                        }}
                    ]
                }}
                """}
            ])

            # Prepare the variables for the template
            variables = method_context_data

            logger.info(f"Method description prompt variables: {variables}")

            # Create and invoke the chain
            chain = prompt | self.llm
            response = chain.invoke(variables)
            
            if response:
                # Extract the content from the response
                if hasattr(response, 'content'):
                    content = response.content
                    # Clean up the content by removing any extra quotes or formatting
                    content = content.strip().strip('"').strip()
                    try:
                        # Parse the JSON response
                        description_data = json.loads(content)
                        return {
                            "method_name": method_name,
                            "package": method_context.get('package', ''),
                            "signature": method_context.get('signature', ''),
                            "description": description_data.get('purpose', ''),
                            "parameters": description_data.get('parameters', [])
                        }
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON response for method {method_name}: {e}")
                        return None
                else:
                    logger.error(f"Unexpected response type for method {method_name}: {type(response)}")
                    return None
            else:
                logger.error(f"Failed to get response for method {method_name}")
                return None
        except Exception as e:
            logger.error(f"Failed to extract description for method {method_context.get('method_name', 'unknown')}: {e}")
            return None

    def extract_all_knowledge(self, parsed_java_data: list[dict], parsed_pom_data: dict | None, output_file: str = "final_extracted_knowledge.json") -> dict:
        """
        Extract all knowledge from parsed Java data and pom.xml using LLM.
        
        Args:
            parsed_java_data: List of dictionaries containing parsed Java file data
            parsed_pom_data: Dictionary containing parsed pom.xml data
            output_file: Path to the output JSON file where incremental results will be saved
            
        Returns:
            Dictionary containing extracted knowledge about the project
        """
        try:
            # Initialize final data structure
            if not os.path.exists(output_file):
                final_extracted_data = {
                    "projectOverview": "",
                    "modules": []
                }
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(final_extracted_data, f, indent=2, ensure_ascii=False)
            else:
                with open(output_file, 'r', encoding='utf-8') as f:
                    final_extracted_data = json.load(f)
            
            # Extract project purpose from pom.xml
            if parsed_pom_data:
                final_extracted_data["projectOverview"] = self._extract_project_purpose(parsed_java_data, parsed_pom_data)
            
            # Process each Java file
            for file_data in parsed_java_data:
                class_name = file_data.get('class_name')
                if not class_name:
                    logger.warning(f"Skipping file with no class name: {file_data}")
                    continue
                    
                logger.info(f"Processing class: {class_name}")
                
                # Extract class description
                class_description = self._extract_class_description(file_data)
                if class_description:
                    logger.info(f"Successfully extracted description for class {class_name}")
                    
                    # Process all methods in the class
                    methods = file_data.get('methods', [])
                    extracted_methods = []
                    
                    for method in methods:
                        method_name = method.get('name')
                        if not method_name:
                            logger.warning(f"Skipping method with no name in class {class_name}")
                            continue
                            
                        logger.info(f"Processing method: {method_name} in class {class_name}")
                        
                        # Prepare method context
                        method_context = {
                            "class_name": class_name,
                            "package": file_data.get("package_name", ""),
                            "method_name": method_name,
                            "signature": method.get("signature", ""),
                            "annotations": method.get("method_annotations", []),
                            "parameters": method.get("parameters", []),
                            "methods": extracted_methods
                        }
                        
                        # Extract method data
                        method_data = self._extract_method_data(method_context, file_data)
                        if method_data:
                            extracted_methods.append(method_data)
                            logger.info(f"Successfully extracted data for method {method_name} in class {class_name}")
                        else:
                            logger.warning(f"Failed to extract data for method {method_name} in class {class_name}")
                            
                    # Add class to modules list
                    final_extracted_data["modules"].append({
                        "name": class_name,
                        "description": class_description.get("purpose", ""),
                        "methods": extracted_methods
                    })
                    logger.info(f"Added class {class_name} to modules list with {len(extracted_methods)} methods")
                    
                    # Save current state to file
                    try:
                        # Use a temporary file for atomic write
                        temp_file = f"{output_file}.tmp"
                        with open(temp_file, 'w', encoding='utf-8') as temp_f:
                            json.dump(final_extracted_data, temp_f, indent=2, ensure_ascii=False)
                        
                        # Verify the contents of the temporary file
                        try:
                            with open(temp_file, 'r', encoding='utf-8') as verify_f:
                                temp_file_contents = json.load(verify_f)
                                logger.info(f"Temporary file contents verified: {json.dumps(temp_file_contents, indent=2)}")
                        except Exception as e:
                            logger.error(f"Failed to verify temporary file contents: {e}")
                        
                        # Replace the original file with the temporary file
                        os.replace(temp_file, output_file)
                        logger.info(f"Successfully saved incremental update for class {class_name} to {output_file}")
                        logger.info(f"Final data structure: {json.dumps(final_extracted_data, indent=2)}")
                    except Exception as e:
                        logger.error(f"Failed to save incremental update for class {class_name}: {e}")
                        logger.debug(f"Data being saved: {json.dumps(final_extracted_data, indent=2)}")
                        raise
                else:
                    logger.error(f"Failed to extract description for class {class_name}")
                    # Add class to final data and save incrementally
                    # First check if this class already exists in modules
                    existing_class = next((module for module in final_extracted_data["modules"] if module["name"] == class_name), None)
                    
                    if existing_class:
                        # If class exists, update its data
                        existing_class["description"] = "Failed to extract description"
                        existing_class["methods"] = []
                        logger.info(f"Updated existing class {class_name} in modules list")
                    else:
                        # If class doesn't exist, append it
                        final_extracted_data["modules"].append({
                            "name": class_name,
                            "description": "Failed to extract description",
                            "methods": []
                        })
                        logger.info(f"Added new class {class_name} to modules list")
                    
                    # Save current state to file
                    try:
                        # Use a temporary file for atomic write
                        temp_file = f"{output_file}.tmp"
                        with open(temp_file, 'w', encoding='utf-8') as f:
                            json.dump(final_extracted_data, f, indent=2, ensure_ascii=False)
                        # Replace the original file with the temporary file
                        os.replace(temp_file, output_file)
                        logger.info(f"Successfully saved incremental update (failed) for class {class_name} to {output_file}")
                        logger.debug(f"Final data structure: {json.dumps(final_extracted_data, indent=2)}")
                    except Exception as e:
                        logger.error(f"Failed to save incremental update (failed) for class {class_name}: {e}")
                        logger.debug(f"Data being saved: {json.dumps(final_extracted_data, indent=2)}")
                        raise
            
            # Clean up any temporary files
            try:
                temp_file = f"{output_file}.tmp"
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.info(f"Removed temporary file: {temp_file}")
            except Exception as cleanup_error:
                logger.error(f"Failed to clean up temporary file: {cleanup_error}")
            
            return final_extracted_data
            
        except Exception as e:
            logger.error(f"Error in extract_all_knowledge: {e}")
            raise

    def _extract_first_json(self, text: str) -> Optional[dict]:
        """
        Extract the first valid JSON object from a mixed text response.
        Handles cases where JSON is embedded in text, in code blocks, or with extra characters.
        
        Args:
            text: The mixed text containing JSON
            
        Returns:
            The first valid JSON object found, or None if no valid JSON is found
        """
        try:
            # Try to find JSON-like patterns and extract them
            json_patterns = [
                r'```json(.*?)```',  # JSON in code blocks
                r'({.*?})',          # JSON object
                r'\[(.*?)\]'       # JSON array
            ]
            
            # Try each pattern
            for pattern in json_patterns:
                matches = re.finditer(pattern, text, re.DOTALL)
                for match in matches:
                    try:
                        # Clean up the match
                        content = match.group(1).strip()
                        # Remove any extra quotes or backticks
                        content = content.strip('`"\'')
                        # Try to parse as JSON
                        return json.loads(content)
                    except json.JSONDecodeError:
                        continue
            
            # If no JSON found in patterns, try parsing the whole text
            try:
                return json.loads(text.strip())
            except json.JSONDecodeError:
                pass
                
            return None
        except Exception as e:
            logger.error(f"Error extracting JSON: {e}")
            return None

    def _extract_method_data(self, method_context: dict, class_context: dict) -> Optional[dict]:
        """
        Extract knowledge about a single method using the LLM.
        
        Args:
            method_context: Dictionary containing method information and context
            class_context: Dictionary containing class context
            
        Returns:
            Dictionary containing extracted method knowledge or None if extraction failed
        """
        try:
            method_name = method_context.get('method_name', 'unknown')
            
            # Debug logging
            logger.info(f"\n=== Processing method {method_name} ===")
            logger.debug(f"Method Context: {json.dumps(method_context, indent=2)}")
            logger.debug(f"Context Length: {len(method_context.get('body', ''))} characters")
            
            # Check cache first if enabled
            if self.cache_enabled and self.cache:
                with self.cache:
                    cached_response = self.cache.get(method_name, method_context.get('body', ''))
                    if cached_response:
                        logger.info(f"Found cached response for method {method_name}")
                        return cached_response
            
            # Create prompt and parser
            # Format parameters and annotations for better readability
            formatted_params = ", ".join([
                f"{param['name']}: {param['type']}" for param in method_context.get('parameters', [])
            ])
            formatted_annotations = ", ".join(method_context.get('annotations', []))
            
            # Format body to be more readable
            body_lines = method_context.get('body', '').split('\n')
            formatted_body = '\n'.join([f"    {line}" for line in body_lines if line.strip()])
            
            # Create the prompt template with proper LangChain message format
            prompt = ChatPromptTemplate.from_messages([
                {"role": "system", "content": "You are an expert Java developer analyzing a method's purpose, complexity, and usage."},
                {"role": "user", "content": """Analyze the following method and provide a detailed analysis:
                
                Method Details:
                Class: {class_name}
                Package: {package}
                Method Name: {method_name}
                Signature: {signature}
                Return Type: {return_type}
                Parameters: {parameters}
                Annotations: {annotations}
                
                Method Body:
                {body}
                
                Please provide a JSON response with the following structure:
                {{
                    "name": "{method_name}",
                    "signature": "{signature}",
                    "description": "Brief description of what the method does",
                    "complexity": "LOW | MEDIUM | HIGH"
                }}
                """}
            ])
            
            # Prepare the variables for the template
            variables = {
                "class_name": method_context.get('class_name', ''),
                "package": method_context.get('package', ''),
                "method_name": method_context.get('method_name', ''),
                "signature": method_context.get('signature', ''),
                "return_type": method_context.get('return_type', 'void'),
                "parameters": formatted_params,
                "annotations": formatted_annotations,
                "body": formatted_body
            }

            logger.info(f"Prompt: {prompt.messages}")
            
            # Try multiple times with different prompts if needed
            for attempt in range(1, self.max_retries + 1):
                try:
                    logger.info(f"\n=== Attempt {attempt}/{self.max_retries} for method {method_name} ===")
                    logger.info(f"Prompt: {prompt.messages}")  # Log the actual prompt being sent
                    logger.debug(f"Variables being passed: {variables}")
                    
                    response_data = (prompt | self.llm).invoke(variables)
                    if response_data:
                        logger.debug(f"LLM Response Type: {type(response_data)}")
                        if hasattr(response_data, 'content'):
                            logger.debug(f"LLM Response Content: {response_data.content}")
                            content = response_data.content
                            try:
                                # Extract JSON from mixed response
                                description_data = self._extract_first_json(content)
                                if description_data:
                                    logger.debug(f"Parsed JSON: {json.dumps(description_data, indent=2)}")
                                
                                    # Map description to purpose since that's what we expect in the output
                                    description_data['purpose'] = description_data.get('description', f"Summary for {method_name}")
                                
                                    # Save to cache if enabled
                                    if self.cache_enabled and self.cache:
                                        with self.cache:
                                            self.cache.save(method_name, method_context.get('body', ''), description_data)
                                
                                    # Save intermediate output
                                    output_context = {
                                        "method_name": method_name,
                                        "class_name": method_context.get('class_name', ''),
                                        "signature": method_context.get('signature', ''),
                                        "raw_content": method_context.get('body', ''),
                                        "response_data": description_data
                                    }
                                
                                    intermediate_output_file = f"intermediate_output_{method_name}.json"
                                    with open(intermediate_output_file, 'w', encoding='utf-8') as f:
                                        json.dump(output_context, f, indent=2, ensure_ascii=False)
                                        logger.info(f"Saved intermediate output to: {intermediate_output_file}")
                                
                                    # Return structured response
                                    return {
                                        "name": method_name,
                                        "signature": method_context.get('signature', ''),
                                        "purpose": description_data['purpose'],  # Now guaranteed to exist
                                        "complexity": description_data.get('complexity', 'MEDIUM'),
                                        "dependencies": description_data.get('dependencies', []),
                                        "usage": description_data.get('usage', ""),
                                        "exceptions": description_data.get('exceptions', [])
                                    }
                                else:
                                    logger.error(f"Failed to extract JSON from response for method {method_name}")
                                    return None
                            except json.JSONDecodeError as e:
                                logger.error(f"JSON decode error: {e}\nResponse content: {content}")
                                raise
                        else:
                            logger.error(f"Response has no content attribute: {response_data}")
                            return None
                    else:
                        logger.error("LLM returned None response")
                        return None
                except json.JSONDecodeError as e:
                    logger.error(f"Attempt {attempt}/{self.max_retries} failed: Invalid JSON response: {e}")
                    if attempt == self.max_retries:
                        raise
                except Exception as e:
                    logger.error(f"Attempt {attempt}/{self.max_retries} failed: {e}")
                    if attempt == self.max_retries:
                        raise
        except Exception as e:
            logger.error(f"Failed to extract knowledge for method {method_name}: {e}")
            return None

    def _extract_project_purpose(self, parsed_java_data: list[dict], parsed_pom_data: dict | None) -> str:
        """
        Extract project purpose from pom.xml data.
        
        Args:
            parsed_java_data: List of dictionaries containing parsed Java file data
            parsed_pom_data: Dictionary containing parsed pom.xml data
            
        Returns:
            String containing the extracted project purpose
        """
        try:
            # Get project information from pom.xml
            project_info = {
                "project_name": parsed_pom_data.get('project_name', 'N/A'),
                "description": parsed_pom_data.get('description', 'N/A'),
                "group_id": parsed_pom_data.get('group_id', 'N/A'),
                "artifact_id": parsed_pom_data.get('artifact_id', 'N/A')
            }
            
            # Create the prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert software architect tasked with summarizing a Java project."),
                ("user", """Analyze the following project information and provide a concise description of the project's purpose:
                
                Project Information:
                Project Name: {project_name}
                Description: {description}
                Group ID: {group_id}
                Artifact ID: {artifact_id}
                
                Please provide a clear and concise description of what this project does.
                """.format(**project_info))
            ])

            # Create and invoke the chain
            chain = prompt | self.llm
            response = chain.invoke({})
            logger.info(f"Project purpose response from LLM: {response}")
            if response:
                # Extract the content from the response
                if hasattr(response, 'content'):
                    content = response.content
                    # Clean up the content by removing any extra quotes or formatting
                    content = content.strip().strip('"').strip()
                    return content
                else:
                    logger.error(f"Unexpected response type: {type(response)}")
                    return str(response)
            else:
                logger.error("Failed to get response for project purpose extraction")
                return "Failed to extract project purpose. Please check the logs for details."
        except Exception as e:
            logger.error(f"Error extracting project purpose: {e}")
            return f"Error extracting project purpose: {str(e)}"

# Example Usage (for testing KnowledgeExtractor independently, assuming JSON exists)
if __name__ == "__main__":
    extractor = KnowledgeExtractor()
    
    if Path(PARSED_JAVA_DATA_FILE).exists():
        parsed_data = extractor.load_parsed_java_data()
        
        # Dummy pom data for independent testing if you don't run main.py
        # In main.py, this will come from parser.py
        dummy_pom_data = {
            "name": "Sakila Project (Dummy)",
            "description": "A dummy project for testing purposes.",
            "groupId": "com.dummy",
            "artifactId": "dummy-app",
            "version": "1.0.0",
            "dependencies": [{"artifactId": "spring-boot-starter-web"}]
        }

        if parsed_data:
            extracted_knowledge = extractor.extract_all_knowledge(parsed_data, dummy_pom_data) # Pass dummy pom data
            
            print("\n--- Sample Extracted Knowledge Structure ---")
            print(json.dumps(extracted_knowledge, indent=2))
            
            final_output_json_file = "final_extracted_knowledge.json"
            with open(final_output_json_file, 'w', encoding='utf-8') as f:
                json.dump(extracted_knowledge, f, indent=2)
            print(f"Final extracted knowledge structure saved to: {final_output_json_file}")
        else:
            print("No parsed data to process.")
    else:
        print(f"'{PARSED_JAVA_DATA_FILE}' not found. Please run main.py (Phase 1 & 2) first.")