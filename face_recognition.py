from deepface import DeepFace
from PIL import Image
import deepface.DeepFace
import numpy as np
import cv2


def detect_face(img: np.ndarray)-> tuple[np.ndarray,list[dict]]:
    # Detect faces from image
    faces = DeepFace.extract_faces(img, detector_backend='opencv')
    # Draw bounding box around faces
    for face in faces:
        facial_area = face["facial_area"]
        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), int(img.shape[0]/150))
    return img,faces