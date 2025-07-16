import cv2
import os
import numpy as np

# Configuration
mask_folder = "Data/train/Mask"  # Your source folder
output_folder = "Data/train/Mask_Converted"  # Output folder

# Create output directory
os.makedirs(output_folder, exist_ok=True)

# Get sorted list of mask files to maintain sequence
mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith(".png") and f.startswith("mask_")])

for idx, mask_file in enumerate(mask_files):
    # Read mask as grayscale (single channel)
    mask = cv2.imread(os.path.join(mask_folder, mask_file), cv2.IMREAD_GRAYSCALE)

    # Convert to binary (0 or 1)
    mask_binary = np.where(mask > 128, 1, 0).astype(np.uint8)

    # Generate sequential filename with leading zeros
    output_filename = f"mask_0{idx:03d}.png"
    output_path = os.path.join(output_folder, output_filename)

    # Save as 1-bit PNG (values 0 and 1)
    cv2.imwrite(output_path, mask_binary * 255)  # Scale to 0/255 for visibility

    # Verify saved image
    saved_img = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
    print(f"Converted {mask_file} -> {output_filename} | Unique values: {np.unique(saved_img)}")

print(f"\nSuccess! Converted {len(mask_files)} masks to 1-bit grayscale PNGs.")