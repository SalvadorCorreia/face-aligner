import cv2
import mediapipe as mp
import numpy as np

# --- Setting ---
image_path = "raw_photos/20200928_135012.jpg"
frame_height = 1920
frame_wight = 1080
crop_factor = 4

# --- Mediapipe Setup ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Load image
image = cv2.imread(image_path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # --- Extract Points ---
        h, w, _ = image.shape
        left_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
        right_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
        nose_tip = np.array([landmarks[1].x * w, landmarks[1].y * h])

        # --- Compute Centroid ---
        centroid = (left_eye + right_eye + nose_tip) / 3

        # --- Draw Triangle & Centroid for Visualization ---
        vis = image.copy()
        cv2.circle(vis, tuple(left_eye.astype(int)), 5, (0, 255, 0), -1)
        cv2.circle(vis, tuple(right_eye.astype(int)), 5, (0, 255, 0), -1)
        cv2.circle(vis, tuple(nose_tip.astype(int)), 5, (0, 255, 0), -1)
        cv2.circle(vis, tuple(centroid.astype(int)), 7, (0, 0, 255), -1)
        cv2.line(vis, tuple(left_eye.astype(int)), tuple(right_eye.astype(int)), (255, 0, 0), 2)
        cv2.line(vis, tuple(left_eye.astype(int)), tuple(nose_tip.astype(int)), (255, 0, 0), 2)
        cv2.line(vis, tuple(right_eye.astype(int)), tuple(nose_tip.astype(int)), (255, 0, 0), 2)

        # Show
        cv2.namedWindow("Landmark Triangle", cv2.WINDOW_NORMAL)  # make window resizable
        cv2.imshow("Landmark Triangle", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No face detected!")

