
from deepface import DeepFace
import numpy as np
from PIL import Image
from deepface.modules import preprocessing
from transformers import AutoTokenizer, SiglipTextModel , SiglipVisionModel ,AutoImageProcessor
import time
from deepface.detectors.DetectorWrapper import build_model


# Preload model for faster inference, (once this function is called the model is a global variable)
face_detection_model_name = 'retinaface'
face_detection_model = build_model(face_detection_model_name)
# cold start it to load in the gpu
DeepFace.extract_faces(np.zeros((64,64,3)), detector_backend=face_detection_model_name,enforce_detection=False)

def detect_face(img: np.ndarray,confidence_threshold: float = 0.95)-> list[dict]:
    # if the image has an alpha channel, remove it
    if img.shape[2] == 4:
        img = img[:,:,:3]
    # Detect faces from image
    faces = DeepFace.extract_faces(img, detector_backend=face_detection_model_name,enforce_detection=False)
    faces = [face for face in faces if face["confidence"] > confidence_threshold]
    return faces


# Preload the face recognition model for faster inference, (once this function is called the model is a global variable)
face_embedding_model_name = 'Facenet512'
face_embedding_model = DeepFace.build_model(face_embedding_model_name)
# cold start it to load in the gpu
DeepFace.represent(np.zeros((64,64,3)), model_name=face_embedding_model_name, normalization='base',detector_backend='skip')

def calculate_face_embedding(faces)-> np.ndarray:
    target_size = face_embedding_model.input_shape 
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
    
    embeddings: np.ndarray = face_embedding_model.model(preproccessed_faces).numpy()
    return embeddings

# PreLoad the image model for faster inference
image_processor = AutoImageProcessor.from_pretrained("google/siglip-base-patch16-256-multilingual")
image_model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-256-multilingual")
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
image_model.to(device) # type: ignore

def calculate_images_embedding(images: list[str])-> list[np.ndarray]:
    print("Starting to calculate embeddings")
    batch_size = 16
    loaded_images = [Image.open(image) for image in images]
    # Remove alpha channel if it exists
    for i in range(len(loaded_images)):
        if loaded_images[i].mode == "RGBA":
            loaded_images[i] = loaded_images[i].convert("RGB")
    embeddings = []
    for i in range(0, len(images), batch_size):
        end_indx = i+batch_size if i+batch_size < len(images) else len(images)
        inputs = image_processor(images=loaded_images[i:end_indx], return_tensors="pt")
        inputs = inputs.to(device)
        outputs = image_model(**inputs) # type: ignore
        logits = outputs.pooler_output
        batch_embeddings = logits.cpu().detach().numpy()
        embeddings.extend(batch_embeddings)
        # Free memory
        del outputs
        del logits
        del batch_embeddings
        del inputs
    return embeddings

text_model = SiglipTextModel.from_pretrained("google/siglip-base-patch16-256-multilingual")
tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-256-multilingual")
text_model.to(device) # type: ignore

def calculate_text_embedding(text: str) -> np.ndarray:
    inputs = tokenizer(text,padding="max_length", return_tensors="pt")
    inputs = inputs.to("cuda")
    outputs = text_model(**inputs) # type: ignore
    logits = outputs.pooler_output
    search_text_embedding = logits[0].cpu().detach().numpy()
    return search_text_embedding
