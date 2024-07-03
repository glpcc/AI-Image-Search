import clickhouse_connect 
import numpy as np
from dotenv import load_dotenv
import os


def extract_face_name(face_embeddings: np.ndarray,threshold = 0.3)-> list[tuple[int,str] | None]:
    # TODO Change default user and password 
    client = clickhouse_connect.get_client(host="localhost",port=8123,user="user",password="apasswordtochange")
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
                    FROM ai_image_search.face_data
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
