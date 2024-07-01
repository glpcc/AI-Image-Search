import clickhouse_connect 
import numpy as np
from dotenv import load_dotenv
import os
def extract_face_name(face_embeddings: list[np.ndarray])-> list[str | None]:
    # TODO Change default user and password 
    client = clickhouse_connect.get_client(host="localhost",port=8123,user="user",password="apasswordtochange")
    face_names = []
    # Get the nearest neighbors for each face embedding
    for embedding in face_embeddings:
        if len(embedding) != 512:
            raise ValueError("Embedding should be of size 512")
        
        result = client.query(f'''SELECT
                        face_name,
                        min(cosineDistance(
                            %(embd)s,
                            face_embedding
                        )) AS confidence
                    FROM ai_image_search.face_data
                    GROUP BY face_name''', {'embd': embedding.tolist()})
        
        # If there are no results, we can say that the face is not recognized
        if len(result.result_rows) == 0:
            face_names.append(None)
            continue

        # Get the name of the face
        face_name,confidence = result.result_rows[0]

        # If the confidence is greater than 0.5, we can say that the face is recognized
        if confidence < 0.3:
            face_names.append(face_name)
            continue

        face_names.append(None)
    return face_names

a = extract_face_name([np.array([1,2,3])])