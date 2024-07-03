import clickhouse_connect 
import numpy as np
from dotenv import load_dotenv
import os
import random

def store_image(image: str)-> tuple[int,str,np.ndarray]:
    # TODO Change default user and password 
    client = clickhouse_connect.get_client(host="localhost",port=8123,user="user",password="apasswordtochange")
    result = []
    # TODO Calculate the image embedding
    image_embedding = np.random.rand(512)
    # Generate a 64bit id for the image
    new_image_id = random.randint(0,2**64)
    # Insert the image in the database
    client.insert(table="image_data",database="ai_image_search",data=[(new_image_id,image,image_embedding)])

    return (new_image_id,image,image_embedding)
