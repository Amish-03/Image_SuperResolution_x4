import kagglehub
import os
import shutil

def download_dataset():
    print("Downloading DIV2K dataset...")
    # Download latest version
    path = kagglehub.dataset_download("soumikrakshit/div2k-high-resolution-images")
    
    print(f"Dataset downloaded to: {path}")
    
    # In a real scenario, we might move this or just verify it exists.
    # For now, we will print the structure so we know how to load it.
    print("Listing downloaded files:")
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for f in files[:5]: # Print first 5 files only
            print(f'{subindent}{f}')
        if len(files) > 5:
            print(f'{subindent}... ({len(files)-5} more files)')

if __name__ == "__main__":
    download_dataset()
