import cv2
import os
import argparse
import numpy as np
import re
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

def contains_black_border(image_crop, threshold=10):
    edges = [
        image_crop[0, :],    
        image_crop[-1, :],   
        image_crop[:, 0],    
        image_crop[:, -1]    
    ]
    for edge in edges:
        black_pixels = np.sum(np.all(edge <= [5, 5, 5], axis=-1))
        if black_pixels > threshold:
            return True
    return False

def apply_blur_fill(frame, blur_kernel=151):
    h_orig, w_orig = frame.shape[:2]
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return frame
        
    c = max(contours, key=cv2.contourArea)
    
    solid_mask = np.zeros_like(gray)
    cv2.drawContours(solid_mask, [c], -1, 255, thickness=cv2.FILLED)
    
    erosion_kernel = np.ones((5, 5), np.uint8)
    eroded_mask = cv2.erode(solid_mask, erosion_kernel, iterations=2)
    
    x, y, w, h = cv2.boundingRect(eroded_mask)
    if w == 0 or h == 0:
        return frame
        
    valid_crop = frame[y:y+h, x:x+w]
    bg = cv2.resize(valid_crop, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
    
    ksize = blur_kernel if blur_kernel % 2 != 0 else blur_kernel + 1
    bg = cv2.GaussianBlur(bg, (ksize, ksize), 0)
    
    feathered_mask = cv2.GaussianBlur(eroded_mask, (31, 31), 0)
    
    alpha = feathered_mask.astype(float) / 255.0
    alpha = np.expand_dims(alpha, axis=2) 
    
    final_frame = (frame * alpha + bg * (1.0 - alpha)).astype(np.uint8)
    
    return final_frame

def get_date_from_file(filename, filepath):
    """Attempts to extract a YYYYMMDD date from the filename. Falls back to file creation time."""
    # Look for a standard 8-digit date string (e.g., 20200928)
    match = re.search(r'(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])', filename)
    
    if match:
        year, month, day = match.groups()
        date_obj = datetime(int(year), int(month), int(day))
        return date_obj.strftime("%b %d, %Y") # Outputs: "Sep 28, 2020"
    else:
        # Fallback: Check the OS modification time of the file
        mtime = os.path.getmtime(filepath)
        date_obj = datetime.fromtimestamp(mtime)
        return date_obj.strftime("%b %Y") # Just Month/Year if we are guessing

def overlay_date_text(frame, date_text):
    """Draws white text with a thick black shadow in the bottom right corner."""
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.2
    thickness = 2
    
    # Calculate the size of the text so we can align it to the right side
    (text_w, text_h), _ = cv2.getTextSize(date_text, font, font_scale, thickness)
    
    # Position: Bottom Right corner with a 40-pixel padding
    frame_h, frame_w = frame.shape[:2]
    x = frame_w - text_w - 40
    y = frame_h - 40
    
    # Draw the black drop shadow (offset by 3 pixels, thicker)
    cv2.putText(frame, date_text, (x + 3, y + 3), font, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    
    # Draw the white text on top
    cv2.putText(frame, date_text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return frame

def main():
    parser = argparse.ArgumentParser(description="Compile aligned photos into a Timelapse Video.")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second (default: 15)")
    parser.add_argument("--mode", type=str, choices=['vanilla', 'strict-crop', 'blur-fill'], default='vanilla')
    parser.add_argument("--crop-width", type=int, default=800, help="Width of strict crop")
    parser.add_argument("--crop-height", type=int, default=800, help="Height of strict crop")
    parser.add_argument("--blur-kernel", type=int, default=151, help="Strength of the background blur (default: 151)")
    
    # NEW ARGUMENT HERE
    parser.add_argument("--show-date", action="store_true", help="Overlay the date in the bottom right corner")
    
    args = parser.parse_args()

    INPUT_DIR = "aligned_photos"
    OUTPUT_DIR = "videos"
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    valid_extensions = ('.png', '.jpg', '.jpeg')
    images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_extensions)]
    images.sort() 

    if not images:
        print(f"Error: No valid images found in '{INPUT_DIR}'.")
        return

    first_frame = cv2.imread(os.path.join(INPUT_DIR, images[0]))
    if first_frame is None:
        print("Error reading the first image.")
        return
        
    orig_h, orig_w, _ = first_frame.shape

    if args.mode == 'strict-crop':
        video_w, video_h = args.crop_width, args.crop_height
        center_x = orig_w // 2
        center_y = int(orig_h * 0.45)
        
        x1 = max(0, center_x - (video_w // 2))
        x2 = min(orig_w, center_x + (video_w // 2))
        y1 = max(0, center_y - (video_h // 2))
        y2 = min(orig_h, center_y + (video_h // 2))
    else:
        video_w, video_h = orig_w, orig_h

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"timelapse_{args.mode}_{args.fps}fps_{timestamp}.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, args.fps, (video_w, video_h))

    processed_count = 0
    discarded_count = 0

    for image_name in tqdm(images, desc=f"Encoding ({args.mode})", unit="frame"):
        img_path = os.path.join(INPUT_DIR, image_name)
        frame = cv2.imread(img_path)
        
        if frame is None:
            continue

        # 1. Apply formatting modes
        if args.mode == 'strict-crop':
            crop = frame[y1:y2, x1:x2]
            if contains_black_border(crop):
                discarded_count += 1
                continue
            final_frame = crop
            
        elif args.mode == 'blur-fill':
            final_frame = apply_blur_fill(frame, args.blur_kernel)
            
        else:
            final_frame = frame
            
        # 2. Overlay Date (if requested)
        if args.show_date:
            date_text = get_date_from_file(image_name, img_path)
            final_frame = overlay_date_text(final_frame, date_text)
            
        # 3. Write to video
        video_writer.write(final_frame)
        processed_count += 1

    video_writer.release()
    
    print("\n" + "="*45)
    print("         VIDEO COMPILATION COMPLETE")
    print("="*45)
    print(f" Output File : {output_path}")
    print(f" Mode        : {args.mode.upper()}")
    print(f" Date Overlay: {'ENABLED' if args.show_date else 'DISABLED'}")
    print(f" Resolution  : {video_w}x{video_h}")
    print(f" Framerate   : {args.fps} FPS")
    print("-" * 45)
    print(f" Images Kept : {processed_count}")
    if args.mode == 'strict-crop':
        print(f" Discarded   : {discarded_count} (Due to black borders)")
    print("="*45 + "\n")

if __name__ == "__main__":
    main()
