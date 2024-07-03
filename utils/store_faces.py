import clickhouse_connect 
import numpy as np
from dotenv import load_dotenv
import os
import random
from PIL import Image

def store_face(faces: list[dict], faces_names_and_ids: list[tuple[int,str] | None],embeddings: np.ndarray,image_id: int)-> list[str | None]:
    # TODO Change default user and password 
    client = clickhouse_connect.get_client(host="localhost",port=8123,user="user",password="apasswordtochange")
    result = []
    for face,face_name,embedding in zip(faces,faces_names_and_ids,embeddings):
        if face_name is not None:
            result.append(face_name)
            # Add the image to the appeared images column
            client.insert(table="image_faces",database="ai_image_search",data=[(image_id,face_name[0])])
            continue       
        

        # TODO: Change this path to a more general one
        temp_folder = "C:\\Users\\gonza\\AppData\\Local\\Temp\\gradio\\faces\\"

        # Generate a 64bit id for the face
        new_face_id = random.randint(0,2**64)
        # Save the face in the file system
        face_path = temp_folder + str(new_face_id) + ".png"
        # The incoming face is a numpy array with shape (height,width,channels)
        img = Image.fromarray(np.uint8(face['face'][:,:,::-1]*255),mode="RGB")
        img.save(face_path)
        # Insert the face in the database
        client.insert(table="face_data",database="ai_image_search",data=[(new_face_id,"Not Named",embedding,face_path)])
        result.append([new_face_id,"Not Named"])

    return result
