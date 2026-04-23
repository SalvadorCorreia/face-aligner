import cv2
import os
import numpy as np
from tqdm import tqdm

def main():
    INPUT_DIR = "aligned_photos"
    
    # Check if directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Directory '{INPUT_DIR}' not found.")
        return

    valid_extensions = ('.png', '.jpg', '.jpeg')
    images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_extensions)]

    if not images:
        print(f"Error: No valid images found in '{INPUT_DIR}'.")
        return

    print(f"Analyzing {len(images)} images to find optimal crop dimensions...\n")

    safe_widths = []
    safe_heights = []

    for image_name in tqdm(images, desc="Scanning image boundaries", unit="img"):
        img_path = os.path.join(INPUT_DIR, image_name)
        img = cv2.imread(img_path)
        
        if img is None:
            continue

        h, w, _ = img.shape
        
        # The exact anchor point used in runner.py
        anchor_x = w // 2
        anchor_y = int(h * 0.45)

        # Convert to grayscale to easily find the black [0,0,0] space
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Threshold: Any pixel strictly > 0 becomes 255 (white)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        
        # Get coordinates of all non-black pixels
        coords = cv2.findNonZero(thresh)
        
        if coords is not None:
            # Get the strict bounding box of the real image data
            x_min, y_min, box_w, box_h = cv2.boundingRect(coords)
            x_max = x_min + box_w
            y_max = y_min + box_h
            
            # To keep the face centered, the crop must be symmetrical around the anchor.
            # We find the shortest distance from the anchor to the edge of the black void
            # and multiply by 2 to get the maximum safe width/height for this specific image.
            max_safe_w = 2 * min(anchor_x - x_min, x_max - anchor_x)
            max_safe_h = 2 * min(anchor_y - y_min, y_max - anchor_y)
            
            # Ensure we don't accidentally calculate a negative or zero dimension
            if max_safe_w > 0 and max_safe_h > 0:
                safe_widths.append(max_safe_w)
                safe_heights.append(max_safe_h)

    if not safe_widths:
        print("Could not calculate bounds. Check if images are completely black.")
        return

    # --- Data Processing & Statistics ---
    safe_widths = np.array(safe_widths)
    safe_heights = np.array(safe_heights)

    # Calculate percentiles
    w_100 = int(np.min(safe_widths))
    h_100 = int(np.min(safe_heights))
    
    w_95 = int(np.percentile(safe_widths, 5))
    h_95 = int(np.percentile(safe_heights, 5))
    
    w_90 = int(np.percentile(safe_widths, 10))
    h_90 = int(np.percentile(safe_heights, 10))

    median_w = int(np.median(safe_widths))
    median_h = int(np.median(safe_heights))

    # --- Print Report ---
    print("\n" + "="*50)
    print("             CROP ANALYSIS REPORT")
    print("="*50)
    print(" To guarantee absolutely ZERO black borders, use ")
    print(" the 100% threshold. However, sacrificing just 5%")
    print(" of your worst outlier photos usually allows for")
    print(" a much larger, higher-resolution video.")
    print("-" * 50)
    print(f" 100% Retained (Smallest Outlier) : {w_100}x{h_100}")
    print(f"  95% Retained (Recommended)      : {w_95}x{h_95}")
    print(f"  90% Retained (Aggressive)       : {w_90}x{h_90}")
    print("-" * 50)
    print(f" Median Safe Size                 : {median_w}x{median_h}")
    print("="*50 + "\n")
    
    print("Example Command:")
    print(f"python scripts/make_video.py --mode strict-crop --crop-width {w_95} --crop-height {h_95}\n")

if __name__ == "__main__":
    main()
