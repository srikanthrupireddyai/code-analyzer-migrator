import os
from dotenv import load_dotenv
from preprocessor import PreProcessor
from parser import JavaCodeParser
from knowledge_extractor import KnowledgeExtractor
from pathlib import Path
import json

# Debug .env file location and content
env_path = Path(__file__).parent / '.env'
print(f"\n=== .env File Debug ===")
print(f".env file path: {env_path}")
if env_path.exists():
    print(f".env file exists: Yes")
    with open(env_path, 'r') as f:
        print(".env file contents:")
        print(f.read())
else:
    print(f".env file exists: No")

# Clear any existing environment variable
if 'OPENAI_API_KEY' in os.environ:
    print(f"Clearing existing OPENAI_API_KEY: {os.environ['OPENAI_API_KEY']}")
    del os.environ['OPENAI_API_KEY']

# Load environment variables
load_dotenv()

# Debug environment variables
print("\n=== Environment Variable Debug ===")
print(f"Loaded environment variables: {os.environ.get('OPENAI_API_KEY')} (length: {len(os.environ.get('OPENAI_API_KEY', ''))})")
print("=== End Debug ===\n")

# Validate API key
api_key = os.environ.get('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Debug environment variables
print("\n=== Environment Variable Debug ===")
print(f"Loaded environment variables: {os.environ.get('OPENAI_API_KEY')}")
print("=== End Debug ===\n")

# --- Configuration for the entire pipeline ---
GITHUB_REPO_URL = "https://github.com/janjakovacevic/SakilaProject.git"
LOCAL_REPO_BASE_DIR = "SakilaProject_Cloned" # Directory where the repo will be cloned

def run_migration_pipeline():
    """
    Orchestrates the Java codebase analysis and preparation pipeline.
    """
    print("--- Starting AI-Assisted Code Migration Pipeline ---")

    # Phase 1: Codebase Acquisition and Initial Scan
    print("\n[Phase 1: Preprocessing]")
    preprocessor = PreProcessor(GITHUB_REPO_URL, LOCAL_REPO_BASE_DIR)
    
    try:
        # Set clean_clone to True for a fresh start, False to reuse existing clone
        preprocessor.clone_repository(clean_clone=False) # Change to True for fresh start if needed
        java_file_paths = preprocessor.find_java_files()
        pom_file_path = preprocessor.find_pom_xml() # Find pom.xml
    except Exception as e:
        print(f"Error during Phase 1: {e}")
        return # Stop if Phase 1 fails

    if not java_file_paths:
        print("No Java files found to process. Exiting pipeline.")
        return

    # Phase 2: Code Reading and Basic Parsing
    print("\n[Phase 2: Parsing Java Files and pom.xml]")
    java_parser = JavaCodeParser()
    parsed_java_data = java_parser.parse_all_java_files(java_file_paths)
    
    parsed_pom_data = None
    if pom_file_path:
        parsed_pom_data = java_parser.parse_pom_xml(pom_file_path)

    if not parsed_java_data:
        print("No Java files were successfully parsed. Exiting pipeline.")
        return

    print("\n--- Pipeline Intermediate Results (Phase 1 & 2) ---")
    print(f"Total Java files found in Phase 1: {len(java_file_paths)}")
    print(f"Total Java files successfully parsed in Phase 2: {len(parsed_java_data)}")
    print(f"pom.xml parsed: {parsed_pom_data is not None}")

    # Save the parsed Java data to a JSON file as an intermediate output
    parsed_java_output_file = "parsed_java_code_structure.json"
    with open(parsed_java_output_file, 'w', encoding='utf-8') as f:
        json.dump(parsed_java_data, f, indent=2)
    print(f"Parsed Java code structure saved to: {parsed_java_output_file}")

    # Save the parsed pom.xml data if available
    if parsed_pom_data:
        parsed_pom_output_file = "parsed_pom_data.json"
        with open(parsed_pom_output_file, 'w', encoding='utf-8') as f:
            json.dump(parsed_pom_data, f, indent=2)
        print(f"Parsed pom.xml data saved to: {parsed_pom_output_file}")

    # Phase 3: LLM Integration and Knowledge Extraction
    print("\n[Phase 3: LLM Knowledge Extraction]")
    # You can specify "gpt-4" here if you prefer to start with it, as discussed earlier.
    # knowledge_extractor = KnowledgeExtractor(llm_model_name="gpt-4")
    knowledge_extractor = KnowledgeExtractor() # Uses default "gpt-4o"
    
    # Pass both parsed Java data and parsed pom data
    final_extracted_knowledge = knowledge_extractor.extract_all_knowledge(parsed_java_data, parsed_pom_data)

    print("\n--- Pipeline Final Results (Phase 3) ---")
    final_output_json_file = "final_extracted_knowledge.json"
    with open(final_output_json_file, 'w', encoding='utf-8') as f:
        json.dump(final_extracted_knowledge, f, indent=2)
    print(f"Final extracted knowledge structure saved to: {final_output_json_file}")

    print("\n--- AI-Assisted Code Migration Pipeline Complete (Phases 1-3) ---")

if __name__ == "__main__":
    run_migration_pipeline()