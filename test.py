from deepface import DeepFace
from deepface.detectors.DetectorWrapper import build_model
from PIL import Image
import numpy as np
from deepface.modules import preprocessing
from transformers import AutoTokenizer, SiglipTextModel , SiglipVisionModel ,AutoImageProcessor
import time

# Preload the face r


# Preload model for faster inference, (once this function is called the model is a global variable)
model_name = 'retinaface'
model = build_model(model_name)
# cold start it to load in the gpu
DeepFace.extract_faces(np.zeros((64,64,3)), detector_backend=model_name,enforce_detection=False)

def detect_face(img: np.ndarray)-> list[dict]:
    # Detect faces from image
    faces = DeepFace.extract_faces(img, detector_backend=model_name,enforce_detection=False)

    return faces