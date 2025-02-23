import os
import shutil
import random

# Comprehensive Shark Species List
SHARK_SPECIES = [
    'Great White Shark',
    'Tiger Shark',
    'Hammerhead Shark',
    'Whale Shark',
    'Whitetip Reef Shark',
    'Blacktip Reef Shark',
    'Bull Shark',
    'Nurse Shark',
    'Mako Shark',
    'Sand Tiger Shark'
]

# Set dataset paths
source_dir = "C:/Users/kemel/.cache/kagglehub/datasets/shark-species"
dest_dir = "data/processed"

def create_dataset_structure():
    """Create directory structure for dataset."""
    print("🏗️ Creating dataset directory structure...")
    
    # Create base directories
    for split in ["train", "val"]:
        for species in SHARK_SPECIES:
            # Create safe directory names by replacing spaces
            safe_species = species.replace(' ', '_')
            path = os.path.join(dest_dir, split, safe_species)
            os.makedirs(path, exist_ok=True)
            print(f"Created folder: {path}")

def process_images():
    """Process and split images for each species."""
    print("🖼️ Processing images...")
    
    for species in SHARK_SPECIES:
        # Create safe directory names
        safe_species = species.replace(' ', '_')
        
        # Source and destination paths
        species_source = os.path.join(source_dir, safe_species)
        
        # Check if source directory exists
        if not os.path.exists(species_source):
            print(f"⚠️ WARNING: No images found for {species}")
            continue
        
        # List all images
        images = [f for f in os.listdir(species_source) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        
        if not images:
            print(f"⚠️ WARNING: No images found in {species_source}")
            continue
        
        print(f"📸 Found {len(images)} images for {species}")
        
        # Shuffle images
        random.shuffle(images)
        
        # Split into train and validation
        train_split = int(0.8 * len(images))
        
        # Copy images
        for i, img in enumerate(images):
            src_path = os.path.join(species_source, img)
            
            # Determine split (train or val)
            split = "train" if i < train_split else "val"
            
            # Destination path
            dest_path = os.path.join(dest_dir, split, safe_species, img)
            
            # Copy image
            shutil.copy(src_path, dest_path)
            
            if i % 50 == 0:  # Print progress periodically
                print(f"✅ Copied {img} → {split}/{safe_species}")

def main():
    print("🦈 SharkSense AI - Dataset Generator")
    
    # Verify source directory exists
    if not os.path.exists(source_dir):
        print(f"❌ ERROR: Source directory not found: {source_dir}")
        return
    
    # Create directory structure
    create_dataset_structure()
    
    # Process and split images
    process_images()
    
    print("🎉 Dataset processing completed!")

if __name__ == "__main__":
    main()