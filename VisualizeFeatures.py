import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern
from skimage.filters import gabor_kernel
from scipy import ndimage
from skimage.filters import gabor

def compare_real_vs_fake_features(real_image_path, fake_image_path):
    """
    Compare feature extraction between real and fake images
    """
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    
    for idx, (img_path, label) in enumerate([(real_image_path, 'REAL'), 
                                               (fake_image_path, 'FAKE')]):
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        col = idx * 2
        
        # Original
        axes[0, col].imshow(img_rgb)
        axes[0, col].set_title(f'{label} - Original', fontweight='bold')
        axes[0, col].axis('off')
        
        # HOG
        fd, hog_image = hog(img_gray, orientations=9, pixels_per_cell=(16, 16),
                            cells_per_block=(2, 2), visualize=True)
        axes[1, col].imshow(hog_image, cmap='hot')
        axes[1, col].set_title(f'{label} - HOG')
        axes[1, col].axis('off')
        
        axes[1, col+1].bar(range(50), fd[:50])
        axes[1, col+1].set_title(f'{label} - HOG Histogram')
        
        # LBP
        lbp = local_binary_pattern(img_gray, P=8, R=2, method='nri_uniform')
        axes[2, col].imshow(lbp, cmap='gray')
        axes[2, col].set_title(f'{label} - LBP')
        axes[2, col].axis('off')
        
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 60), density=True, range=(0, 60))
        axes[2, col+1].bar(range(len(lbp_hist)), lbp_hist)
        axes[2, col+1].set_title(f'{label} - LBP Histogram')
        
        # Color
        colors = ('r', 'g', 'b')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
            axes[3, col].plot(hist, color=color, alpha=0.7)
        axes[3, col].set_title(f'{label} - Color Histogram')
        axes[3, col].legend(['R', 'G', 'B'])
        
        # Gabor
        gabor_features = np.zeros_like(img_gray, dtype=np.float32)
        for theta in [0, np.pi/2]:
            #theta_val = theta / 4. * np.pi
            for frequency in [0.1, 0.4]:
                kernel = np.real(gabor_kernel(frequency, theta=theta))
                filtered = ndimage.convolve(img_gray, kernel, mode='wrap')
                gabor_features += np.abs(filtered)
        
        gabor_features = gabor_features / gabor_features.max()
        axes[3, col+1].imshow(gabor_features, cmap='jet')
        axes[3, col+1].set_title(f'{label} - Gabor Response')
        axes[3, col+1].axis('off')

    # Hide all unused axes in the 4x4 grid
    # If the axis doesn't have an image plotted on it
    for ax in axes.flatten():
        if not ax.images:  
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('real_vs_fake_comparison.png', dpi=150, bbox_inches='tight')
    print("Comparison saved to: real_vs_fake_comparison.png")
    plt.show()

compare_real_vs_fake_features('C:/Users/(SrvAzr)AristotelisG/Desktop/ML/MSc-in-AI-ML-Deepfake-Image-Recognition/Datasets/train/real/12220.jpg', 'C:/Users/(SrvAzr)AristotelisG/Desktop/ML/MSc-in-AI-ML-Deepfake-Image-Recognition/Datasets/train/fake/0GWN1MW3Y1.jpg')
