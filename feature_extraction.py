import cv2
import os
import numpy as np
from skimage.feature import local_binary_pattern, hog
from skimage.filters import gabor
from skimage import color
import matplotlib.pyplot as plt
import pandas as pd

# --- SETTINGS ---
FOLDER_PATH = [
    {'load':'C:/Users/(SrvAzr)AristotelisG/Downloads/machine_learning_ask/Dataset/train/fake', 'save':'C:/Users/(SrvAzr)AristotelisG/Downloads/machine_learning_ask/Features/train/fake/train_fake_all_features.npz'},
    {'load':'C:/Users/(SrvAzr)AristotelisG/Downloads/machine_learning_ask/Dataset/train/real', 'save': 'C:/Users/(SrvAzr)AristotelisG/Downloads/machine_learning_ask/Features/train/real/train_real_all_features.npz'},
    {'load':'C:/Users/(SrvAzr)AristotelisG/Downloads/machine_learning_ask/Dataset/valid/fake', 'save': 'C:/Users/(SrvAzr)AristotelisG/Downloads/machine_learning_ask/Features/valid/fake/valid_fake_all_features.npz'},
    {'load':'C:/Users/(SrvAzr)AristotelisG/Downloads/machine_learning_ask/Dataset/valid/real', 'save': 'C:/Users/(SrvAzr)AristotelisG/Downloads/machine_learning_ask/Features/valid/real/valid_real_all_features.npz'},
    {'load':'C:/Users/(SrvAzr)AristotelisG/Downloads/machine_learning_ask/Dataset/test/fake', 'save':'C:/Users/(SrvAzr)AristotelisG/Downloads/machine_learning_ask/Features/test/fake/test_fake_all_features.npz'},
    {'load':'C:/Users/(SrvAzr)AristotelisG/Downloads/machine_learning_ask/Dataset/test/real', 'save':'C:/Users/(SrvAzr)AristotelisG/Downloads/machine_learning_ask/Features/test/real/test_real_all_features.npz'}
] 

IMAGE_SIZE = (128, 128) # Resizing helps control the massive HOG vector size

def hog_extract_features(img_path):
    # Load and Resize
    image = cv2.imread(img_path)    
    image = cv2.resize(image, IMAGE_SIZE)        
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
    image = cv2.resize(image, IMAGE_SIZE)             
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # LBP Features (Histogram)
    lbp = local_binary_pattern(gray, P=8, R=2, method='nri_uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 60), density=True)

    # Combine all into 1D array
    return np.concatenate([lbp_hist])

def color_extract_features(img_path):
     # Load and Resize
    image = cv2.imread(img_path)    
    image = cv2.resize(image, IMAGE_SIZE)   

    # Color Features (Mean/Std per BGR channel)
    color_feats = np.concatenate([np.mean(image, axis=(0,1)), np.std(image, axis=(0,1))])

     # Combine all into 1D array
    return np.concatenate([color_feats])

def gabor_extract_features(img_path):
     # Load and Resize
    image = cv2.imread(img_path)    
    image = cv2.resize(image, IMAGE_SIZE)
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
    for filename in os.listdir(fol['load']):        
        full_path = os.path.join(fol['load'], filename)
        
        print(f"Processed: {filename}")

        hog_vector = hog_extract_features(full_path)
        lbp_vector = lbp_extract_features(full_path)
        color_vector = color_extract_features(full_path)
        gabor_vector = gabor_extract_features(full_path)
        
        hog_list.append(hog_vector)
        lbp_list.append(lbp_vector)
        color_list.append(color_vector)
        gabor_list.append(gabor_vector)
        names_list.append(filename) 

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
