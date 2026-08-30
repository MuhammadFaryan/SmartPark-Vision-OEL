import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import time
import glob

# Load YOLOv8n
model = YOLO('yolov8n.pt')

# Load the first image from the dataset
image_paths = glob.glob("dataset/**/*.jpg", recursive=True)
test_image_path = image_paths[0] 
original_img = cv2.imread(test_image_path)
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

# Get dynamic image dimensions
h, w = original_img.shape[:2]

# Roll Number Custom Threshold
custom_threshold = 883 

# DYNAMIC ROIs: Forces the boxes onto the bottom 40% of the image (the road)
rois = [
    # Slot 1 (Left Highway Lanes)
    np.array([
        [int(w * 0.30), int(h * 0.60)], # Top-left
        [int(w * 0.45), int(h * 0.60)], # Top-right
        [int(w * 0.40), int(h * 0.95)], # Bottom-right
        [int(w * 0.15), int(h * 0.95)]  # Bottom-left
    ], np.int32),
    
    # Slot 2 (Right Highway Lanes)
    np.array([
        [int(w * 0.55), int(h * 0.60)], # Top-left
        [int(w * 0.70), int(h * 0.60)], # Top-right
        [int(w * 0.95), int(h * 0.95)], # Bottom-right
        [int(w * 0.60), int(h * 0.95)]  # Bottom-left
    ], np.int32)
]

def show_img(title, img, cmap=None):
    plt.figure(figsize=(8, 5))
    plt.title(title)
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.show()

# Display Original with ROIs
temp_img = original_img.copy()
cv2.polylines(temp_img, rois, True, (0, 255, 255), 4) # Thickened the lines to 4 so they are clearly visible
show_img("Original Image with Fixed ROIs", temp_img)
