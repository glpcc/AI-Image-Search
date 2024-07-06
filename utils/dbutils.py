import clickhouse_connect 
import clickhouse_connect.common
from clickhouse_connect.driver.client import Client
import numpy as np
from dotenv import load_dotenv
import os
import random
from PIL import Image

clickhouse_client = None
def get_client() -> Client:
    global clickhouse_client
    if clickhouse_client is None:
        # TODO Change default user and password 
        client = clickhouse_connect.get_client(host="localhost",port=8123,user="user",password="apasswordtochange")
        return client
    else:
        return clickhouse_client

def get_faces_to_name()-> list[tuple[int,Image.Image]]:
    client = get_client()
    dbresult = client.query(f'''SELECT
                    id,
                    example_image_name
                FROM ai_image_search.face_data FINAL
                WHERE face_name = 'Not Named'
                
                ''')
    # TODO Handle the file not found exception
    result = [ (id,Image.open(example_image_name)) for id,example_image_name in dbresult.result_rows]
    return result


def store_face(faces: list[dict], faces_names_and_ids: list[tuple[int,str] | None],embeddings: np.ndarray,image_id: int)-> list[str | None]:
    client = get_client()
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

def store_image(image: str)-> tuple[int,str,np.ndarray]:
    client = get_client()
    result = []
    # TODO Calculate the image embedding
    image_embedding = np.random.rand(512)
    # Generate a 64bit id for the image
    new_image_id = random.randint(0,2**64)
    # Insert the image in the database
    client.insert(table="image_data",database="ai_image_search",data=[(new_image_id,image,image_embedding)])

    return (new_image_id,image,image_embedding)

def store_face_name(face_id: int,face_name: str):
    client = get_client()
    # Get all data of the face to then insert a replacement
    dbresult = client.query(f'''SELECT
                    *
                FROM ai_image_search.face_data
                WHERE id = %(face_id)s
                LIMIT 1
                ''', {'face_id': face_id})
    
    client.insert(table="face_data",database="ai_image_search",data=[(face_id,face_name,dbresult.result_rows[0][2],dbresult.result_rows[0][3])])

def extract_face_name(face_embeddings: np.ndarray,threshold = 0.3)-> list[tuple[int,str] | None]:
    client = get_client()
    face_names_and_ids = []
    # Get the nearest neighbors for each face embedding
    for embedding in face_embeddings:
        if len(embedding) != 512:
            raise ValueError("Embedding should be of size 512")
        
        result = client.query(f'''SELECT
                        id,
                        face_name,
                        cosineDistance(
                            %(embd)s,
                            face_embedding
                        ) AS confidence
                    FROM ai_image_search.face_data FINAL
                    ORDER BY confidence asc 
                    LIMIT 1
                    ''', {'embd': embedding.tolist()})
        
        # If there are no results, we can say that the face is not recognized
        if len(result.result_rows) == 0:
            face_names_and_ids.append(None)
            continue

        # Get the name of the face
        id,face_name,confidence = result.result_rows[0]

        # If the confidence is greater than 0.5, we can say that the face is recognized
        if confidence < threshold:
            face_names_and_ids.append((id,face_name))
            continue

        face_names_and_ids.append(None)
    return face_names_and_ids

# TODO Remove this function
def clean_db():
    client = get_client()
    client.query("TRUNCATE TABLE ai_image_search.face_data")
    client.query("TRUNCATE TABLE ai_image_search.image_data")
    client.query("TRUNCATE TABLE ai_image_search.image_faces")