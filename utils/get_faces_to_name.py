import clickhouse_connect 
import numpy as np
from dotenv import load_dotenv
import os
import random
from PIL import Image

def get_faces_to_name()-> list[tuple[int,Image.Image]]:
    # TODO Change default user and password 
    client = clickhouse_connect.get_client(host="localhost",port=8123,user="user",password="apasswordtochange")
    dbresult = client.query(f'''SELECT
                    id,
                    example_image_name
                FROM ai_image_search.face_data
                WHERE face_name = 'Not Named'
                ''')
    # TODO Handle the file not found exception
    result = [ (id,Image.open(example_image_name)) for id,example_image_name in dbresult.result_rows]
    print(result)
    return result
