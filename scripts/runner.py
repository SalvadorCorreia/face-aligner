import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path

# --- Settings ---
INPUT_DIR = "raw_photos"
OUTPUT_DIR = "aligned_photos"
TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920

# Create output folder if it doesn't exist
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh

def get_tier1_matrix(left_eye, right_eye, nose, target_w, target_h):
    """Tier 1: 3-Point Primary (Eyes + Nose)"""
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    
    dist = np.sqrt(dx**2 + dy**2)
    desired_dist = target_w * 0.3
    scale = desired_dist / dist if dist > 0 else 1.0
    
    # Calculate Centroid
    centroid = (left_eye + right_eye + nose) / 3.0
    
    M = cv2.getRotationMatrix2D(tuple(centroid), angle, scale)
    # Target the centroid to the horizontal middle, slightly above vertical middle
    M[0, 2] += (target_w * 0.5) - centroid[0]
    M[1, 2] += (target_h * 0.45) - centroid[1] 
    
    return M

def get_tier2_matrix(left_eye, right_eye, target_w, target_h):
    """Tier 2: 2-Point Fallback (Eyes Only for Masks/Distortions)"""
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    
    dist = np.sqrt(dx**2 + dy**2)
    desired_dist = target_w * 0.3
    scale = desired_dist / dist if dist > 0 else 1.0
    
    eye_center = (left_eye + right_eye) / 2.0
    
    M = cv2.getRotationMatrix2D(tuple(eye_center), angle, scale)
    M[0, 2] += (target_w * 0.5) - eye_center[0]
    M[1, 2] += (target_h * 0.4) - eye_center[1]
    
    return M

def process_images():
    # --- Logging Counters ---
    tier_1_count = 0
    tier_2_count = 0
    fail_count = 0
    total_images = 0

    print("Starting alignment process...")

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        for filename in os.listdir(INPUT_DIR):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            total_images += 1
            input_path = os.path.join(INPUT_DIR, filename)
            output_path = os.path.join(OUTPUT_DIR, filename)
            
            image = cv2.imread(input_path)
            if image is None:
                continue
                
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_image)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                h, w, _ = image.shape
                
                left_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
                right_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
                nose_tip = np.array([landmarks[1].x * w, landmarks[1].y * h])
                
                # --- Structural Sanity Check ---
                # In OpenCV, Y-coordinates increase as you go DOWN the screen.
                # Therefore, a valid nose MUST have a higher Y value than the eyes.
                eye_y_avg = (left_eye[1] + right_eye[1]) / 2.0
                
                # Check if nose is safely below the eyes (by at least 2% of image height)
                if nose_tip[1] > eye_y_avg + (h * 0.02): 
                    M = get_tier1_matrix(left_eye, right_eye, nose_tip, TARGET_WIDTH, TARGET_HEIGHT)
                    tier_1_count += 1
                else:
                    M = get_tier2_matrix(left_eye, right_eye, TARGET_WIDTH, TARGET_HEIGHT)
                    tier_2_count += 1
                
                aligned_image = cv2.warpAffine(image, M, (TARGET_WIDTH, TARGET_HEIGHT))
                cv2.imwrite(output_path, aligned_image)
            else:
                fail_count += 1
                print(f"Failed to detect face in: {filename}")

    # --- Print Execution Logs ---
    print("\n" + "="*35)
    print("      ALIGNMENT RUN COMPLETE")
    print("="*35)
    print(f" Total Images Scanned : {total_images}")
    print(f" Tier 1 (Eyes + Nose) : {tier_1_count}")
    print(f" Tier 2 (Eyes Only)   : {tier_2_count}")
    print(f" Failed (No Face)     : {fail_count}")
    print("="*35 + "\n")

if __name__ == "__main__":
    process_images()
