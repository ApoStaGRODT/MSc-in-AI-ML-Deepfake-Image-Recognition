import cv2
import os
import numpy as np
from skimage.feature import local_binary_pattern, hog
from skimage.filters import gabor
from skimage import color
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import time

# Parse command line arguments
parser = argparse.ArgumentParser(description='Feature extraction with flexible paths')
parser.add_argument('--source', type=str, required=True, help='Source directory containing the dataset')
parser.add_argument('--output', type=str, default='.', help='Output directory (default: current directory)')
args = parser.parse_args()

# Convert to Path objects
source_path = Path(args.source)
output_path = Path(args.output)

# Define the folder structure
FOLDER_PATH = [
    {
        'load': source_path / 'train/fake',
        'save': output_path / 'train/fake/train_fake_all_features.npz'
    },
    {
        'load': source_path / 'train/real',
        'save': output_path / 'train/real/train_real_all_features.npz'
    },
    {
        'load': source_path / 'valid/fake',
        'save': output_path / 'valid/fake/valid_fake_all_features.npz'
    },
    {
        'load': source_path / 'valid/real',
        'save': output_path / 'valid/real/valid_real_all_features.npz'
    },
    {
        'load': source_path / 'test/fake',
        'save': output_path / 'test/fake/test_fake_all_features.npz'
    },
    {
        'load': source_path / 'test/real',
        'save': output_path / 'test/real/test_real_all_features.npz'
    }
]

# Create output directories
for folder in FOLDER_PATH:
    folder['save'].parent.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (128, 128) # Resizing helps control the massive HOG vector size

def hog_extract_features(img_path):
    # Load and Resize
    image = cv2.imread(img_path)    
    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # HOG Features
    # Increased pixels_per_cell to (16,16) to keep vector size manageable
    hog_feats = hog(gray, orientations=9, pixels_per_cell=(16, 16), 
                    cells_per_block=(2, 2), feature_vector=True)
    
    # Combine all into 1D array
    return np.concatenate([hog_feats])

def lbp_extract_features(img_path):
     # Load and Resize
    image = cv2.imread(img_path)   
    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # LBP Features (Histogram)
    lbp = local_binary_pattern(gray, P=8, R=2, method='nri_uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 60), density=True)

    # Combine all into 1D array
    return np.concatenate([lbp_hist])

def color_extract_features(img_path):
     # Load and Resize
    image = cv2.imread(img_path)    
    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    # Color Features (Mean/Std per BGR channel)
    color_feats = np.concatenate([np.mean(image, axis=(0,1)), np.std(image, axis=(0,1))])

     # Combine all into 1D array
    return np.concatenate([color_feats])

def gabor_extract_features(img_path):
     # Load and Resize
    image = cv2.imread(img_path)    
    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gabor Features
    g_feats = []
    for freq in [0.1, 0.4]: # Reduced count for speed
        for theta in [0, np.pi/2]:
            filt_real, _ = gabor(gray, frequency=freq, theta=theta)
            g_feats.extend([filt_real.mean(), filt_real.var()])
    
    # Combine all into 1D array
    return np.concatenate([g_feats])


print("Starting extraction...")

for fol in FOLDER_PATH:    
    hog_list, lbp_list, color_list, gabor_list, names_list = [], [], [], [], []

    # Total timers
    total_hog_time = 0
    total_lbp_time = 0
    total_color_time = 0
    total_gabor_time = 0
    folder_start_time = time.time()

    for filename in os.listdir(fol['load']):        
        full_path = os.path.join(fol['load'], filename)

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        #print(f"[{timestamp}] Processing: {filename}")   

        start = time.time()
        hog_vector = hog_extract_features(full_path)
        hog_time = time.time() - start
        total_hog_time += hog_time

        start = time.time()
        lbp_vector = lbp_extract_features(full_path)
        lbp_time = time.time() - start
        total_lbp_time += lbp_time

        start = time.time()
        color_vector = color_extract_features(full_path)
        color_time = time.time() - start
        total_color_time += color_time

        start = time.time()
        gabor_vector = gabor_extract_features(full_path)     
        gabor_time = time.time() - start
        total_gabor_time += gabor_time

        image_total_time = hog_time + lbp_time + color_time + gabor_time
        
        hog_list.append(hog_vector)
        lbp_list.append(lbp_vector)
        color_list.append(color_vector)
        gabor_list.append(gabor_vector)
        names_list.append(filename) 

    # Print summary for this folder
    folder_total_time = time.time() - folder_start_time
    num_images = len(names_list)     
    print("\n" + "="*60)
    print(f"SUMMARY for {fol['load']}")
    print("="*60)
    print(f"Total images processed: {num_images}")
    print(f"Total time: {folder_total_time:.2f}s ({folder_total_time/60:.2f} minutes)")
    print(f"\nTime per method:")
    print(f"  HOG:   {total_hog_time:.2f}s ({total_hog_time/num_images:.4f}s per image) - {total_hog_time/folder_total_time*100:.1f}%")
    print(f"  LBP:   {total_lbp_time:.2f}s ({total_lbp_time/num_images:.4f}s per image) - {total_lbp_time/folder_total_time*100:.1f}%")
    print(f"  Color: {total_color_time:.2f}s ({total_color_time/num_images:.4f}s per image) - {total_color_time/folder_total_time*100:.1f}%")
    print(f"  Gabor: {total_gabor_time:.2f}s ({total_gabor_time/num_images:.4f}s per image) - {total_gabor_time/folder_total_time*100:.1f}%")
    print("="*60 + "\n")    

    # Save
    np.savez_compressed(
        fol['save'], 
        hog=np.array(hog_list), 
        lbp=np.array(lbp_list), 
        color=np.array(color_list), 
        gabor=np.array(gabor_list),
        filenames=np.array(names_list)
    )

    print(f"Success! Saved data for {len(names_list)} images.")

# Print one sample from each feature        
print("\n" + "="*60)
print(f"Sample features from: {names_list[0]}")
print("="*60)
print(f"HOG feature (first 5):     {hog_list[0][:5]}")
print(f"LBP feature (first 5):     {lbp_list[0][:5]}")
print(f"Color feature (first 5):   {color_list[0][:5]}")
print(f"Gabor feature (first 5):   {gabor_list[0][:5]}")
print(f"\nFeature vector shapes:")
print(f"HOG:   {hog_list[0].shape}")
print(f"LBP:   {lbp_list[0].shape}")
print(f"Color: {color_list[0].shape}")
print(f"Gabor: {gabor_list[0].shape}")
print(f"Execution Time -> HOG: {hog_time:.4f}s | LBP: {lbp_time:.4f}s | Color: {color_time:.4f}s | Gabor: {gabor_time:.4f}s | Total: {image_total_time:.4f}s")
print("="*60 + "\n")


