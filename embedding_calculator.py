from deepface import DeepFace
import numpy as np
from PIL import Image
from deepface.modules import preprocessing
import time

# Preload model for faster inference, (once this function is called the model is a global variable)
model_name = 'Facenet512'
model = DeepFace.build_model(model_name)
# cold start it to load in the gpu
DeepFace.represent(np.zeros((64,64,3)), model_name=model_name, normalization='base',detector_backend='skip')

def calculate_embedding(faces)-> np.ndarray:
    target_size = model.input_shape 
    preproccessed_faces = np.empty((len(faces), target_size[0], target_size[1], 3))
    for i,face in enumerate(faces):
        img = face["face"]
        # rgb to bgr
        img = img[:, :, ::-1]

        # resize to expected shape of ml model
        img = preprocessing.resize_image(
            img=img,
            # thanks to DeepId (!)
            target_size=(target_size[1], target_size[0]),
        )

        # custom normalization
        img = preprocessing.normalize_input(img=img, normalization="base")

        preproccessed_faces[i] = img
    
    embeddings: np.ndarray = model.model(preproccessed_faces).numpy()
    return embeddings
