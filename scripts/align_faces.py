import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm

# ----------------- SETTINGS -----------------
input_folder = "Dataset"
output_folder = "photos_aligned"
os.makedirs(output_folder, exist_ok=True)

final_size = 1024          # output frame size (1024x1024)
crop_multiplier = 3.5      # how much larger the crop is than face width
# --------------------------------------------

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# Helper: compute distance between two points
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def align_face(image, landmarks):
    h, w, _ = image.shape

    # Pick left/right eye and face points
    left_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
    right_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
    nose_tip = np.array([landmarks[1].x * w, landmarks[1].y * h])

    # Compute rotation angle
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # Eyes center
    eyes_center = ((left_eye[0] + right_eye[0]) / 2,
                   (left_eye[1] + right_eye[1]) / 2)

    # Rotate image
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1)
    rotated = cv2.warpAffine(image, M, (w, h))

    # Face width
    face_width = distance(left_eye, right_eye)
    crop_size = int(face_width * crop_multiplier)

    # Crop around eyes center
    cx, cy = int(eyes_center[0]), int(eyes_center[1])
    half = crop_size // 2
    x1, y1 = max(cx - half, 0), max(cy - half, 0)
    x2, y2 = x1 + crop_size, y1 + crop_size

    cropped = rotated[y1:y2, x1:x2]

    # Handle borders
    cropped = cv2.copyMakeBorder(
        cropped,
        top=max(0, -y1),
        bottom=max(0, y2 - rotated.shape[0]),
        left=max(0, -x1),
        right=max(0, x2 - rotated.shape[1]),
        borderType=cv2.BORDER_CONSTANT,
        value=[0,0,0]
    )

    # Resize to final size
    return cv2.resize(cropped, (final_size, final_size))

# -------- PROCESS IMAGES --------
failed_images = []

for file in tqdm(os.listdir(input_folder)):
    if not (file.lower().endswith(".jpg") or file.lower().endswith(".png")):
        continue

    path = os.path.join(input_folder, file)
    image = cv2.imread(path)
    if image is None:
        continue

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        aligned = align_face(image, landmarks)
        cv2.imwrite(os.path.join(output_folder, file), aligned)
    else:
        failed_images.append(file)

# Log failed detections
if failed_images:
    print("No face detected in:", failed_images)
else:
    print("All images processed successfully!")

