import cv2
import os
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

def contains_black_border(image_crop, threshold=10):
    """
    Scans the 4 edges of the cropped image. 
    If more than 'threshold' pure black pixels are found on any edge,
    it assumes we hit the empty space created by the alignment warp.
    """
    edges = [
        image_crop[0, :],    # Top edge
        image_crop[-1, :],   # Bottom edge
        image_crop[:, 0],    # Left edge
        image_crop[:, -1]    # Right edge
    ]
    
    for edge in edges:
        # Check how many pixels on this edge are exactly [0, 0, 0]
        black_pixels = np.sum(np.all(edge == [0, 0, 0], axis=-1))
        if black_pixels > threshold:
            return True
            
    return False

def main():
    # --- Setup Command Line Arguments ---
    parser = argparse.ArgumentParser(description="Compile aligned photos into a Timelapse Video.")
    
    # Video settings
    parser.add_argument("--fps", type=int, default=15, help="Frames per second (default: 15)")
    parser.add_argument("--mode", type=str, choices=['vanilla', 'strict-crop'], default='vanilla', 
                        help="Choose 'vanilla' (Option 1) or 'strict-crop' (Option 2)")
    
    # Crop settings (Only used if mode == 'strict-crop')
    parser.add_argument("--crop-width", type=int, default=800, help="Width of strict crop (default: 800)")
    parser.add_argument("--crop-height", type=int, default=800, help="Height of strict crop (default: 800)")
    
    args = parser.parse_args()

    # --- Directories ---
    INPUT_DIR = "aligned_photos"
    OUTPUT_DIR = "videos"
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # --- Fetch and Sort Images ---
    valid_extensions = ('.png', '.jpg', '.jpeg')
    images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_extensions)]
    images.sort() 

    if not images:
        print(f"Error: No valid images found in '{INPUT_DIR}'.")
        return

    # --- Video Setup ---
    first_frame = cv2.imread(os.path.join(INPUT_DIR, images[0]))
    if first_frame is None:
        print("Error reading the first image.")
        return
        
    orig_h, orig_w, _ = first_frame.shape

    # Determine final video dimensions based on mode
    if args.mode == 'strict-crop':
        video_w, video_h = args.crop_width, args.crop_height
        print(f"Mode: Strict Crop. Target Resolution: {video_w}x{video_h}")
        
        # Calculate crop coordinates. 
        # We center the crop horizontally, and put it at 45% vertically 
        # to match where runner.py anchored the face centroid.
        center_x = orig_w // 2
        center_y = int(orig_h * 0.45)
        
        x1 = max(0, center_x - (video_w // 2))
        x2 = min(orig_w, center_x + (video_w // 2))
        y1 = max(0, center_y - (video_h // 2))
        y2 = min(orig_h, center_y + (video_h // 2))
    else:
        video_w, video_h = orig_w, orig_h
        print(f"Mode: Vanilla. Target Resolution: {video_w}x{video_h}")

    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"timelapse_{args.mode}_{args.fps}fps_{timestamp}.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, args.fps, (video_w, video_h))

    # --- Tracking variables for logging ---
    processed_count = 0
    discarded_count = 0

    # --- Compile Video Loop ---
    for image_name in tqdm(images, desc="Encoding Video", unit="frame"):
        img_path = os.path.join(INPUT_DIR, image_name)
        frame = cv2.imread(img_path)
        
        if frame is None:
            continue

        if args.mode == 'strict-crop':
            # 1. Apply the crop
            crop = frame[y1:y2, x1:x2]
            
            # 2. Check for black borders
            if contains_black_border(crop):
                discarded_count += 1
                continue  # Skip this image and move to the next
                
            # 3. If it passes, prepare to write
            final_frame = crop
        else:
            # Vanilla mode
            final_frame = frame
            
        video_writer.write(final_frame)
        processed_count += 1

    # Clean up
    video_writer.release()
    
    # --- Final Logging Output ---
    print("\n" + "="*45)
    print("         VIDEO COMPILATION COMPLETE")
    print("="*45)
    print(f" Output File : {output_path}")
    print(f" Mode        : {args.mode.upper()}")
    print(f" Resolution  : {video_w}x{video_h}")
    print(f" Framerate   : {args.fps} FPS")
    print("-" * 45)
    print(f" Images Kept : {processed_count}")
    if args.mode == 'strict-crop':
        print(f" Discarded   : {discarded_count} (Due to black borders)")
    print("="*45 + "\n")

if __name__ == "__main__":
    main()
