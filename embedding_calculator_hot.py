from deepface import DeepFace
from deepface.basemodels.Facenet import FaceNet128dClient
from deepface.modules import preprocessing
import numpy as np
from PIL import Image
import time

# Load the face recognition model
model: FaceNet128dClient = DeepFace.build_model('Facenet')

def calculate_embedding(faces)-> list[np.ndarray]:
    target_size = model.input_shape
    # Calculate embedding from image
    embeddings = []
    for face in faces:
        img = face["face"]

        # rgb to bgr
        img = img[:, :, ::-1]

        region = face["facial_area"]
        confidence = face["confidence"]

        # resize to expected shape of ml model
        img = preprocessing.resize_image(
            img=img,
            # thanks to DeepId (!)
            target_size=(target_size[1], target_size[0]),
        )
        normalization = 'base'
        # custom normalization
        img = preprocessing.normalize_input(img=img, normalization=normalization)

        embedding = model.forward(img)

        resp_obj = {}
        resp_obj["embedding"] = embedding
        resp_obj["facial_area"] = region
        resp_obj["face_confidence"] = confidence
        embeddings.append(resp_obj)
    return embeddings
