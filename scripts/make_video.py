import cv2
import os
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

def main():
    # --- Setup Command Line Arguments ---
    parser = argparse.ArgumentParser(description="Compile aligned photos into a Timelapse Video.")
    parser.add_argument(
        "--fps", 
        type=int, 
        default=15, 
        help="Frames per second for the output video (default: 15)"
    )
    args = parser.parse_args()

    # --- Directories ---
    INPUT_DIR = "aligned_photos"
    OUTPUT_DIR = "videos"

    # Create output directory if it doesn't exist
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # --- Fetch and Sort Images ---
    valid_extensions = ('.png', '.jpg', '.jpeg')
    images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_extensions)]
    
    # CRITICAL: Images must be sorted alphabetically/chronologically
    images.sort() 

    if not images:
        print(f"Error: No valid images found in '{INPUT_DIR}'.")
        return

    print(f"Found {len(images)} images. Preparing to compile video at {args.fps} FPS...")

    # --- Video Setup ---
    # Read the first image to dynamically get the width and height
    first_image_path = os.path.join(INPUT_DIR, images[0])
    first_frame = cv2.imread(first_image_path)
    if first_frame is None:
        print(f"Error reading the first image: {first_image_path}")
        return
        
    height, width, _ = first_frame.shape

    # Generate a descriptive filename (e.g., timelapse_15fps_20260423_150722.mp4)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"timelapse_{args.fps}fps_{timestamp}.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    # Initialize OpenCV Video Writer
    # 'mp4v' is a highly compatible codec for .mp4 containers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, args.fps, (width, height))

    # --- Compile Video Loop ---
    # Wrap our loop in tqdm for a sleek progress bar
    for image_name in tqdm(images, desc="Encoding Video", unit="frame"):
        img_path = os.path.join(INPUT_DIR, image_name)
        frame = cv2.imread(img_path)
        
        # Write the frame to the video
        if frame is not None:
            video_writer.write(frame)

    # Clean up and save the file
    video_writer.release()
    print("\n" + "="*40)
    print("      VIDEO COMPILATION COMPLETE")
    print("="*40)
    print(f" Output File : {output_path}")
    print(f" Resolution  : {width}x{height}")
    print(f" Framerate   : {args.fps} FPS")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
