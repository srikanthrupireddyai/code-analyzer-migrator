import subprocess
import os
import shutil
from pathlib import Path

class PreProcessor:
    """
    Handles the acquisition of the Java codebase and identification of Java source files.
    """
    def __init__(self, repo_url: str, local_repo_dir_name: str = "SakilaProject_Cloned"):
        self.repo_url = repo_url
        self.local_repo_dir = Path(local_repo_dir_name)

    def clone_repository(self, clean_clone: bool = True):
        """
        Clones a Git repository to the specified local directory.
        If clean_clone is True, removes the directory if it already exists before cloning.
        """
        if self.local_repo_dir.exists():
            if clean_clone:
                print(f"Directory '{self.local_repo_dir}' already exists. Removing and re-cloning...")
                shutil.rmtree(self.local_repo_dir)
            else:
                print(f"Directory '{self.local_repo_dir}' already exists. Skipping cloning.")
                return False # Indicate that cloning was skipped

        print(f"Cloning '{self.repo_url}' into '{self.local_repo_dir}'...")
        try:
            result = subprocess.run(
                ['git', 'clone', self.repo_url, str(self.local_repo_dir)],
                check=True,
                capture_output=True,
                text=True
            )
            print("Cloning successful!")
            # print(result.stdout) # Uncomment for detailed git output
            return True # Indicate successful cloning
        except subprocess.CalledProcessError as e:
            print(f"Error cloning repository: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            raise
        except FileNotFoundError:
            print("Error: 'git' command not found. Please ensure Git is installed and in your system's PATH.")
            raise

    def find_java_files(self) -> list[Path]:
        """
        Recursively finds all .java files within the cloned repository directory.
        Returns a list of pathlib.Path objects.
        """
        if not self.local_repo_dir.exists():
            print(f"Error: Repository directory '{self.local_repo_dir}' not found. Please clone the repository first.")
            return []

        java_files = []
        print(f"Scanning '{self.local_repo_dir}' for Java files...")
        for file_path in self.local_repo_dir.rglob("*.java"):
            java_files.append(file_path) # Store as Path object directly

        print(f"Found {len(java_files)} Java files.")
        return java_files
    
    def find_pom_xml(self) -> Path | None:
        """
        Finds the pom.xml file in the root of the cloned repository.
        Returns a pathlib.Path object if found, otherwise None.
        """
        pom_path = self.local_repo_dir / "pom.xml"
        if pom_path.exists() and pom_path.is_file():
            print(f"Found pom.xml at: {pom_path}")
            return pom_path
        else:
            print(f"pom.xml not found in the root directory: {self.local_repo_dir}")
            return None

# Example Usage (for testing PreProcessor.py independently)
if __name__ == "__main__":
    REPO_URL = "https://github.com/janjakovacevic/SakilaProject.git"
    LOCAL_DIR = "SakilaProject_Cloned"

    preprocessor = PreProcessor(REPO_URL, LOCAL_DIR)
    
    try:
        preprocessor.clone_repository(clean_clone=True) # Set to False after first successful run to save time
        all_java_files = preprocessor.find_java_files()
        pom_file = preprocessor.find_pom_xml()
        
        print("\n--- Preprocessor Output Sample ---")
        print(f"Total Java files found: {len(all_java_files)}")
        print(f"pom.xml found: {pom_file is not None}")
        if pom_file:
            print(f"  Path: {pom_file}")
        print("First 5 identified Java file paths:")
        for i, file_path in enumerate(all_java_files[:5]):
            print(f"- {file_path}")
        if len(all_java_files) > 5:
            print(f"...and {len(all_java_files) - 5} more.")

    except Exception as e:
        print(f"An error occurred during Preprocessor execution: {e}")