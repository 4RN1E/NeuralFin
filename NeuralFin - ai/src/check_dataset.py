import os
import sys

def check_dataset():
    print("Current Working Directory:", os.getcwd())
    print("Python Executable:", sys.executable)
    
    print("\nCurrent Directory Contents:")
    print(os.listdir())
    
    print("\nSearching for data directories:")
    potential_data_dirs = [
        'data/raw/sharks',  # Check the data/raw/sharks folder
        'data/processed',    # Check the data/processed folder
        'data',              # Check the data folder itself
        'C:/Users/kemel/.cache/kagglehub/datasets/larusso94/shark-species/versions/1/sharks'  # Check Kaggle cache
    ]
    
    found_dirs = []
    for data_dir in potential_data_dirs:
        if os.path.exists(data_dir):
            print(f"\nFound directory: {data_dir}")
            found_dirs.append(data_dir)
            
            # List contents of the directory
            try:
                print("Directory contents:")
                print(os.listdir(data_dir))
            except Exception as e:
                print(f"Error listing contents: {e}")
    
    if not found_dirs:
        print("No data directories found!")

if __name__ == "__main__":
    check_dataset()
    input("Press Enter to exit...")  # Prevents immediate window closure