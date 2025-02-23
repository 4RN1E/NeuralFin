import os
import sys

def main():
    print("Project Root Test Script")
    print(f"Python Version: {sys.version}")
    print(f"Current Working Directory: {os.getcwd()}")
    print(f"Script Location: {os.path.abspath(__file__)}")
    
    # Check project structure
    print("\nProject Structure:")
    print("Directories:")
    print(os.listdir())

if __name__ == "__main__":
    main()