# 1. Grayscale
gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
show_img("Grayscale", gray, cmap='gray')

# 2. Gaussian Blur (7x7)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)
show_img("Gaussian Blur (7x7)", blurred, cmap='gray')

# 3. Thresholding Comparison
_, global_thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
show_img("Global Thresholding", global_thresh, cmap='gray')

adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 11, 2)
show_img("Adaptive Thresholding", adaptive_thresh, cmap='gray')

# 4. Morphological Dilation
kernel = np.ones((3,3), np.uint8)
dilated = cv2.dilate(adaptive_thresh, kernel, iterations=1)
show_img("Morphological Dilation", dilated, cmap='gray')

# 5. Pixel Counting & Occupancy Classification
classical_output = original_img.copy()
for i, roi in enumerate(rois):
    mask = np.zeros_like(gray)
    cv2.fillPoly(mask, [roi], 255)
    roi_pixels = cv2.bitwise_and(dilated, dilated, mask=mask)
    
    pixel_count = cv2.countNonZero(roi_pixels)
    
    # Classification using Roll Number Threshold (883)
    status = "Occupied" if pixel_count > custom_threshold else "Free"
    color = (255, 0, 0) if status == "Occupied" else (0, 255, 0) # Red = Occupied, Green = Free
    
    cv2.polylines(classical_output, [roi], True, color, 3)
    # Add dark background for text readability
    cv2.rectangle(classical_output, (roi[0][0], roi[0][1]-25), (roi[0][0]+120, roi[0][1]), (0,0,0), -1)
    cv2.putText(classical_output, f"Slot {i+1}: {pixel_count}px", (roi[0][0]+5, roi[0][1]-8), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

show_img("Classical Pipeline: Occupancy Output", classical_output)
