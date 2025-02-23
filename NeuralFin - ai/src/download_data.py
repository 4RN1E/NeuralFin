import os
import shutil
import kagglehub
import pandas as pd

def download_shark_dataset(dataset_name="larusso94/shark-species", download_dir='../data/downloaded'):
    """
    Download shark dataset from Kaggle and prepare for training
    
    Args:
        dataset_name (str): Kaggle dataset identifier
        download_dir (str): Directory to store downloaded images
    
    Returns:
        str: Path to downloaded dataset
    """
    # Print start of download process
    print(f"Starting dataset download: {dataset_name}")
    
    # Ensure download directory exists
    os.makedirs(download_dir, exist_ok=True)
    
    try:
        # Download dataset
        print("Attempting to download dataset...")
        path = kagglehub.dataset_download(dataset_name)
        print(f"Dataset downloaded successfully to: {path}")
        
        # Organize dataset for training
        organize_dataset(path, download_dir)
        
        return path
    
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Troubleshooting steps:")
        print("1. Ensure you have kagglehub installed (pip install kagglehub)")
        print("2. Verify Kaggle API credentials are set up correctly")
        print("3. Check your internet connection")
        return None

def organize_dataset(source_path, dest_path):
    """
    Organize downloaded dataset into train-friendly structure
    
    Args:
        source_path (str): Path to downloaded dataset
        dest_path (str): Destination path for organized dataset
    """
    # Print start of organization process
    print("Organizing dataset...")
    
    # Clear existing destination directory
    shutil.rmtree(dest_path, ignore_errors=True)
    os.makedirs(dest_path, exist_ok=True)
    
    # Create subdirectories for different classes
    os.makedirs(os.path.join(dest_path, 'shark'), exist_ok=True)
    os.makedirs(os.path.join(dest_path, 'no_shark'), exist_ok=True)
    
    # Find image files in the downloaded dataset
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    
    # Counters for images
    shark_count = 0
    no_shark_count = 0
    
    # Walk through the source directory
    for root, dirs, files in os.walk(source_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                full_path = os.path.join(root, file)
                
                # Basic classification logic 
                # You may need to adjust this based on your specific dataset
                if 'shark' in file.lower():
                    dest_file = os.path.join(dest_path, 'shark', file)
                    shark_count += 1
                else:
                    dest_file = os.path.join(dest_path, 'no_shark', file)
                    no_shark_count += 1
                
                # Copy file
                shutil.copy(full_path, dest_file)
    
    # Print organization results
    print("Dataset organized successfully!")
    print(f"Shark images: {shark_count}")
    print(f"Non-shark images: {no_shark_count}")
    print(f"Total images: {shark_count + no_shark_count}")

def main():
    # Print start of main function
    print("Starting dataset download and organization...")
    
    # Download and organize dataset
    download_shark_dataset()
    
    print("Dataset preparation complete!")

if __name__ == "__main__":
    main()