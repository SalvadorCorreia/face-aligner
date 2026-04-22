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

def get_transformation_matrix(left_eye, right_eye, target_w, target_h):
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
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        for filename in os.listdir(INPUT_DIR):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
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
                
                M = get_transformation_matrix(left_eye, right_eye, TARGET_WIDTH, TARGET_HEIGHT)
                aligned_image = cv2.warpAffine(image, M, (TARGET_WIDTH, TARGET_HEIGHT))
                
                cv2.imwrite(output_path, aligned_image)

if __name__ == "__main__":
    process_images()
