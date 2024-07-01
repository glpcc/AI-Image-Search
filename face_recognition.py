from deepface import DeepFace
from deepface.detectors.DetectorWrapper import build_model
from PIL import Image
import deepface.DeepFace
import numpy as np
import cv2

# Preload model for faster inference, (once this function is called the model is a global variable)
model_name = 'retinaface'
model = build_model(model_name)
# cold start it to load in the gpu
DeepFace.extract_faces(np.zeros((64,64,3)), detector_backend=model_name,enforce_detection=False)

def detect_face(img: np.ndarray)-> tuple[np.ndarray,list[dict]]:
    # Detect faces from image
    faces = DeepFace.extract_faces(img, detector_backend=model_name)
    # Draw bounding box around faces
    for face in faces:
        facial_area = face["facial_area"]
        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), int(img.shape[0]/150))
    return img,faces