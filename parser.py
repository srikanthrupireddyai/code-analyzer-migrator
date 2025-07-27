import javalang
import os
import re
from pathlib import Path
import json # For pretty printing output
import xml.etree.ElementTree as ET # For XML parsing

class JavaCodeParser:
    """
    Parses individual Java files and extracts structured information.
    """
    def __init__(self):
        pass # No specific initialization needed for this class yet

    def _get_method_body(self, file_content: str, method_node: javalang.tree.MethodDeclaration) -> str:
        """
        Heuristic to extract the raw code of a method from file content using AST node positions.
        This is an approximation and might not be perfect for all edge cases.
        """
        try:
            if not hasattr(method_node, 'position') or not hasattr(method_node, 'end_position'):
                return ""  # Fallback if position info is missing

            start_pos = method_node.position
            end_pos = method_node.end_position

            if not start_pos or not end_pos:
                return ""  # Fallback if position info is missing

            lines = file_content.splitlines(keepends=True)

            start_line_idx = start_pos.line - 1
            end_line_idx = end_pos.line - 1

            # Validate indices
            if start_line_idx < 0 or end_line_idx >= len(lines):
                return ""

            method_text_lines = []
            if start_line_idx == end_line_idx:
                # Single line method
                method_text_lines.append(lines[start_line_idx][start_pos.column - 1 : end_pos.column])
            else:
                # Multi-line method
                method_text_lines.append(lines[start_line_idx][start_pos.column - 1:])
                for i in range(start_line_idx + 1, end_line_idx):
                    method_text_lines.append(lines[i])
                method_text_lines.append(lines[end_line_idx][:end_pos.column])

            return "".join(method_text_lines).strip()
        except Exception as e:
            print(f"Warning: Failed to extract method body for {method_node.name}: {e}")
            return ""


    def parse_java_file(self, filepath: Path) -> dict | None:
        """
        Parses a single Java file (Path object) and extracts structured information.
        Returns a dictionary of parsed data or None if parsing fails.
        """
        try:
            parsed_data = {
                "file_path": str(filepath),
                "raw_content": None,
                "category": "Other", # Default category, as discussed
                "package_name": None,
                "class_name": None,
                "class_annotations": [],
                "imports": [],
                "extends": None,
                "implements": [],
                "fields": [],
                "methods": []
            }

            with open(filepath, 'r', encoding='utf-8') as f:
                file_content = f.read()
                parsed_data["raw_content"] = file_content

                try:
                    tree = javalang.parse.parse(file_content)
                except javalang.parser.JavaSyntaxError as e:
                    print(f"Syntax Error in {filepath}: {e}")
                    return None
                except javalang.tokenizer.LexerError as e:
                    print(f"Lexer Error in {filepath}: {e}")
                    return None
                except Exception as e:
                    print(f"AST Parsing Error in {filepath}: {e}")
                    return None

            # Extract Package Name
            if tree.package:
                parsed_data["package_name"] = tree.package.name

            # Extract Imports
            if tree.imports:
                for imp in tree.imports:
                    parsed_data["imports"].append(imp.path)

            # Iterate through types (classes, interfaces, enums)
            # Assuming one main class/interface per file for simplicity
            for path, node in tree.filter(javalang.tree.TypeDeclaration): # Covers ClassDeclaration and InterfaceDeclaration
                if not hasattr(node, 'body') or node.body is None:
                    continue  # Skip if no body is found

                parsed_data["class_name"] = node.name

                # Extract Class Annotations
                if node.annotations:
                    for annotation in node.annotations:
                        annotation_name = annotation.name.member if hasattr(annotation.name, 'member') else annotation.name
                        parsed_data["class_annotations"].append(annotation_name)

                # Extract Extends and Implements
                if hasattr(node, 'extends') and node.extends:
                    # For interfaces, extends is a list of Type objects
                    if isinstance(node.extends, list):
                        parsed_data["extends"] = [ext.name for ext in node.extends]
                    else:
                        parsed_data["extends"] = node.extends.name
                if hasattr(node, 'implements') and node.implements:
                    for impl in node.implements:
                        parsed_data["implements"].append(impl.name)
                
                # Categorize the class (order matters for precedence)
                # First check if it's a repository interface
                if isinstance(node, javalang.tree.InterfaceDeclaration):
                    # For interfaces, check if any extended type is JpaRepository
                    if isinstance(parsed_data["extends"], list):
                        if any("JpaRepository" in ext for ext in parsed_data["extends"]):
                            parsed_data["category"] = "DAO"
                    elif isinstance(parsed_data["extends"], str):
                        if "JpaRepository" in parsed_data["extends"]:
                            parsed_data["category"] = "DAO"
                elif parsed_data["extends"] and "JpaRepository" in parsed_data["extends"]:
                    parsed_data["category"] = "DAO"
                # Then check for specific annotations
                elif any(annotation in parsed_data["class_annotations"] 
                       for annotation in ["RestController", "Controller", "Controller.class"]):
                    parsed_data["category"] = "Controller"
                elif any(annotation in parsed_data["class_annotations"] 
                        for annotation in ["Service", "Service.class"]):
                    parsed_data["category"] = "Service"
                elif any(annotation in parsed_data["class_annotations"] 
                        for annotation in ["Repository", "Repository.class", "JpaRepository", "JpaRepository.class"]):
                    parsed_data["category"] = "DAO"
                elif any(annotation in parsed_data["class_annotations"] 
                        for annotation in ["Entity", "Entity.class"]):
                    parsed_data["category"] = "Entity"
                # Then check package structure
                elif parsed_data["package_name"] and any(
                        parsed_data["package_name"].startswith(prefix)
                        for prefix in ["com.sparta.engineering72.sakilaproject.controller",
                                     "com.sparta.engineering72.sakilaproject.service",
                                     "com.sparta.engineering72.sakilaproject.dao",
                                     "com.sparta.engineering72.sakilaproject.repository"]):
                    # Infer category based on package structure
                    if "controller" in parsed_data["package_name"]:
                        parsed_data["category"] = "Controller"
                    elif "service" in parsed_data["package_name"]:
                        parsed_data["category"] = "Service"
                    elif any(pkg in parsed_data["package_name"] for pkg in ["dao", "repository"]):
                        parsed_data["category"] = "DAO"
                # Then check class name patterns
                elif parsed_data["class_name"]:
                    if parsed_data["class_name"].endswith("Controller"):
                        parsed_data["category"] = "Controller"
                    elif parsed_data["class_name"].endswith("Service"):
                        parsed_data["category"] = "Service"
                    elif parsed_data["class_name"].endswith(("DAO", "Repository")):
                        parsed_data["category"] = "DAO"
                    elif parsed_data["class_name"].endswith("Entity"):
                        parsed_data["category"] = "Entity"
                # Else, it remains "Other" as per default initialization

                # Extract Fields (member variables)
                if node.body:
                    for member in node.body:
                        if isinstance(member, javalang.tree.FieldDeclaration):
                            for declarator in member.declarators:
                                parsed_data["fields"].append({
                                    "name": declarator.name,
                                    "type": member.type.name
                                })

                # Extract Methods
                if node.body:
                    for method in node.body:
                        if isinstance(method, javalang.tree.MethodDeclaration):
                            try:
                                method_signature = f"{method.return_type.name if method.return_type else 'void'} {method.name}(" + \
                                                   ", ".join([f"{param.type.name} {param.name}" for param in method.parameters]) + ")"
                                
                                method_annotations = []
                                if method.annotations:
                                    for annotation in method.annotations:
                                        annotation_name = annotation.name.member if hasattr(annotation.name, 'member') else annotation.name
                                        method_annotations.append(annotation_name)

                                method_body_raw_code = self._get_method_body(file_content, method) # Use the helper method

                                parsed_data["methods"].append({
                                    "name": method.name,
                                    "signature": method_signature,
                                    "method_annotations": method_annotations,
                                    "parameters": [{"name": p.name, "type": p.type.name} for p in method.parameters],
                                    "return_type": method.return_type.name if method.return_type else "void",
                                    "method_body_raw_code": method_body_raw_code
                                })
                            except Exception as e:
                                print(f"Warning: Failed to process method {method.name} in {filepath}: {e}")
                break # Assuming one main class/interface per file for simplicity in Sakila project
            
            return parsed_data

        except Exception as e:
            print(f"An unexpected error occurred while parsing {filepath}: {e}")
            return None

    def parse_pom_xml(self, pom_filepath: Path) -> dict | None:
        """
        Parses a pom.xml file and extracts key project information.
        """
        if not pom_filepath.exists():
            print(f"Error: pom.xml file '{pom_filepath}' not found.")
            return None

        parsed_pom_data = {
            "name": None,
            "description": None,
            "groupId": None,
            "artifactId": None,
            "version": None,
            "dependencies": []
        }

        try:
            tree = ET.parse(pom_filepath)
            root = tree.getroot()

            # XML namespaces can be tricky. Maven POMs typically use a default namespace.
            # We need to register it or use the full qualified name.
            # The default namespace for Maven POM is http://maven.apache.org/POM/4.0.0
            namespace = {'mvn': 'http://maven.apache.org/POM/4.0.0'}

            # Extract basic project info
            parsed_pom_data["name"] = root.find('mvn:name', namespace).text if root.find('mvn:name', namespace) is not None else None
            parsed_pom_data["description"] = root.find('mvn:description', namespace).text if root.find('mvn:description', namespace) is not None else None
            parsed_pom_data["groupId"] = root.find('mvn:groupId', namespace).text if root.find('mvn:groupId', namespace) is not None else None
            parsed_pom_data["artifactId"] = root.find('mvn:artifactId', namespace).text if root.find('mvn:artifactId', namespace) is not None else None
            parsed_pom_data["version"] = root.find('mvn:version', namespace).text if root.find('mvn:version', namespace) is not None else None
            
            # If groupId/artifactId/version are not found at root, check in parent
            if not parsed_pom_data["groupId"] and root.find('mvn:parent/mvn:groupId', namespace) is not None:
                parsed_pom_data["groupId"] = root.find('mvn:parent/mvn:groupId', namespace).text
            if not parsed_pom_data["artifactId"] and root.find('mvn:parent/mvn:artifactId', namespace) is not None:
                parsed_pom_data["artifactId"] = root.find('mvn:parent/mvn:artifactId', namespace).text
            if not parsed_pom_data["version"] and root.find('mvn:parent/mvn:version', namespace) is not None:
                parsed_pom_data["version"] = root.find('mvn:parent/mvn:version', namespace).text


            # Extract dependencies
            dependencies_node = root.find('mvn:dependencies', namespace)
            if dependencies_node is not None:
                for dependency in dependencies_node.findall('mvn:dependency', namespace):
                    dep_info = {
                        "groupId": dependency.find('mvn:groupId', namespace).text if dependency.find('mvn:groupId', namespace) is not None else None,
                        "artifactId": dependency.find('mvn:artifactId', namespace).text if dependency.find('mvn:artifactId', namespace) is not None else None,
                        "version": dependency.find('mvn:version', namespace).text if dependency.find('mvn:version', namespace) is not None else None,
                        "scope": dependency.find('mvn:scope', namespace).text if dependency.find('mvn:scope', namespace) is not None else None
                    }
                    parsed_pom_data["dependencies"].append(dep_info)
            
            print(f"Successfully parsed pom.xml from {pom_filepath}.")
            return parsed_pom_data

        except ET.ParseError as e:
            print(f"Error parsing pom.xml at {pom_filepath}: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred while parsing pom.xml at {pom_filepath}: {e}")
            return None
    
    def parse_all_java_files(self, java_file_paths: list[Path]) -> list[dict]:
        """
        Processes a list of Java file paths (from Preprocessor) and parses each one.
        Returns a list of structured dictionaries for all successfully parsed files.
        """
        parsed_results = []
        print("\n--- Starting Phase 2: Parsing Java Files ---")
        for file_path in java_file_paths:
            print(f"Parsing: {file_path}")
            parsed_data = self.parse_java_file(file_path)
            if parsed_data:
                parsed_results.append(parsed_data)
        print(f"Successfully parsed {len(parsed_results)} out of {len(java_file_paths)} files.")
        return parsed_results

# Example Usage (for testing parser.py independently)
if __name__ == "__main__":
    # This block assumes you've manually cloned the repo and have some paths
    # In main.py, you'd get this list from Preprocessor
    LOCAL_REPO_DIR = Path("SakilaProject_Cloned")
    
    # Dummy list for testing:
    # IMPORTANT: Ensure these files actually exist in your LOCAL_REPO_DIR after cloning
    dummy_java_files = [
        LOCAL_REPO_DIR / "src/main/java/com/sakila/controller/ActorController.java",
        LOCAL_REPO_DIR / "src/main/java/com/sakila/service/ActorService.java",
        LOCAL_REPO_DIR / "src/main/java/com/sakila/dao/ActorDAO.java",
        LOCAL_REPO_DIR / "src/main/java/com/sakila/entity/Actor.java",
        LOCAL_REPO_DIR / "src/main/java/com/sakila/utility/MyUtility.java" # Example for 'Other' category
    ]
    # Filter to ensure they exist for this run
    java_files_to_process = [f for f in dummy_java_files if f.exists()]

    if not java_files_to_process:
        print("No Java files found for independent testing. Please ensure 'SakilaProject_Cloned' exists with files.")
    else:
        parser = JavaCodeParser()
        parsed_data_output = parser.parse_all_java_files(java_files_to_process)

        print("\n--- Sample Parsed Data (First Parsed File) ---")
        if parsed_data_output:
            print(json.dumps(parsed_data_output[0], indent=2))
        
        print("\n--- Categorization Summary ---")
        category_counts = {}
        for data in parsed_data_output:
            category_counts[data["category"]] = category_counts.get(data["category"], 0) + 1
        print(json.dumps(category_counts, indent=2))

        # Test pom.xml parsing
        print("\n--- Testing pom.xml Parsing ---")
        if pom_file_path_test.exists():
            parsed_pom_data = parser.parse_pom_xml(pom_file_path_test)
            if parsed_pom_data:
                print(json.dumps(parsed_pom_data, indent=2))
        else:
            print(f"pom.xml not found at {pom_file_path_test} for independent testing.")