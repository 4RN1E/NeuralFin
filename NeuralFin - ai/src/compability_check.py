import sys
import platform

def check_compatibility():
    print("Python Version Check:")
    print(f"Python Version: {sys.version}")
    print(f"Python Implementation: {platform.python_implementation()}")
    print(f"Python Build: {platform.python_build()}")
    
    print("\nLibrary Versions:")
    try:
        import torch
        print(f"Torch Version: {torch.__version__}")
    except ImportError:
        print("Torch: Not installed")
    
    try:
        import torchvision
        print(f"Torchvision Version: {torchvision.__version__}")
    except ImportError:
        print("Torchvision: Not installed")
    
    try:
        import PIL
        print(f"Pillow Version: {PIL.__version__}")
    except ImportError:
        print("Pillow: Not installed")

if __name__ == "__main__":
    check_compatibility()