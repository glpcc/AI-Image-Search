from deepface import DeepFace
import numpy as np
from PIL import Image
import time

# Preload model for faster inference, (once this function is called the model is a global variable)
model_name = 'Facenet512'
model = DeepFace.build_model(model_name)
# cold start it to load in the gpu
DeepFace.represent(np.zeros((64,64,3)), model_name=model_name, normalization='base',detector_backend='skip')

def calculate_embedding(faces)-> list[np.ndarray]:
    embeddings = []
    for face in faces:
        embd = DeepFace.represent(face["face"], model_name=model_name, normalization='base',detector_backend='skip')
        embeddings.append(embd)
    return embeddings
