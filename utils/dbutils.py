import clickhouse_connect 
import clickhouse_connect.common
from clickhouse_connect.driver.client import Client
import numpy as np
from dotenv import load_dotenv
import os
import random
from PIL import Image

clickhouse_client = clickhouse_connect.get_client(host="localhost",port=8123,user="user",password="apasswordtochange")
def get_client() -> Client:
    if clickhouse_client is None:
        # TODO Change default user and password 
        client = clickhouse_connect.get_client(host="localhost",port=8123,user="user",password="apasswordtochange")
        return client
    else:
        return clickhouse_client

def get_faces_to_name(num_of_needed_appearances: int)-> list[tuple[int,Image.Image]]:
    client = get_client()
    dbresult = client.query(f'''SELECT
                    id,
                    any(example_image_name)
                FROM ai_image_search.face_data JOIN ai_image_search.image_faces on id = face_id
                WHERE face_name = 'Not Named'
                GROUP BY id
                HAVING count() >= %(num_of_needed_appearances)s
                ''', {'num_of_needed_appearances': num_of_needed_appearances})
    # TODO Handle the file not found exception
    result = [ (id,Image.open(example_image_name)) for id,example_image_name in dbresult.result_rows]
    return result


def store_face(faces: list[dict], faces_names_and_ids: list[tuple[int,str] | None],embeddings: np.ndarray,image_id: int)-> list[str]:
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
        client.insert(table="image_faces",database="ai_image_search",data=[(image_id,new_face_id)])
        result.append([new_face_id,"Not Named"])

    return result

def store_images(images: list[str],image_ids: list[int], embeddings: list[np.ndarray])-> None:
    client = get_client()
    all_data = [(image_id,image,embedding) for image,image_id,embedding in zip(images,image_ids,embeddings)]
    # Insert the image in the database
    client.insert(table="image_data",database="ai_image_search",data=all_data)

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

def get_named_faces()-> dict[str,list[int]]:
    client = get_client()
    dbresult = client.query(f'''SELECT
                    id,
                    face_name
                FROM ai_image_search.face_data FINAL
                WHERE face_name != 'Not Named'
                
                ''')
    faces_to_ids = dict()
    # Join the faces with the same name
    for id,face_name in dbresult.result_rows:
        if face_name not in faces_to_ids:
            faces_to_ids[face_name] = []
        faces_to_ids[face_name].append(id)
    return faces_to_ids

def get_images_by_face(faces_ids: list[int],num_repeated_face_names: int)-> list[str]:
    # print(f"Num repeated face names: {num_repeated_face_names}")
    # print(f"Faces ids: {faces_ids}")
    client = get_client()
    dbresult = client.query(f'''
                SELECT
                    any(image_name)
                FROM ai_image_search.image_faces INNER JOIN ai_image_search.image_data on image_id = id
                WHERE face_id in %(faces_ids)s
                GROUP BY image_id
                HAVING uniqExact(face_id) = %(faces_ids_len)s
                ''', {'faces_ids': faces_ids,'faces_ids_len': len(faces_ids) - num_repeated_face_names})
    return [image_name[0] for image_name in dbresult.result_rows] # type: ignore

def extract_face_name(face_embeddings: np.ndarray,threshold = 0.4)-> list[tuple[int,str] | None]:
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

def search_images_by_embbeding(embedding: np.ndarray,limit: int)-> list[tuple[str,float]]:
    client = get_client()
    result = client.query(f'''SELECT
                    image_name,
                    cosineDistance(
                        %(embd)s,
                        image_embedding
                    ) AS score
                FROM ai_image_search.image_data
                ORDER BY score ASC
                LIMIT %(limit)s''', {'embd': embedding.tolist(),'limit': limit})
    return [tuple(image_data) for image_data in result.result_rows]




# TODO Remove this function
def clean_db():
    client = get_client()
    client.query("TRUNCATE TABLE ai_image_search.face_data")
    client.query("TRUNCATE TABLE ai_image_search.image_data")
    client.query("TRUNCATE TABLE ai_image_search.image_faces")