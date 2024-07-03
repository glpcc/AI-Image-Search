import gradio as gr
from utils.face_recognition import detect_face
from utils.embedding_calculator import calculate_embedding
from utils.compare_faces import extract_face_name
from utils.store_faces import store_face
from utils.store_image import store_image
from utils.get_faces_to_name import get_faces_to_name
from PIL import Image
import numpy as np
import time
import cProfile
import pstats

def detect_faces(files: list[str]):
    # with cProfile.Profile() as pr:
    images = []
    total_time = 0
    a = time.time()
    for file in files:
        img = Image.open(file)
        # Detect the faces in the image
        new_img,faces = detect_face(np.array(img))
        # Calculate the embeddings of the faces
        embeddings = calculate_embedding(faces)
        # Get if there is any similar face in the database
        faces_names_and_ids = extract_face_name(embeddings)
        print(faces_names_and_ids)
        # Store the image in the database
        image_id = store_image(file)[0]
        # Store the faces in the database
        faces_names_and_ids = store_face(faces,faces_names_and_ids,embeddings,image_id)
        print(faces_names_and_ids)
        images.append(new_img)
    total_time = time.time() - a
    print(f"Total time taken: {total_time}")
    # stats = pstats.Stats(pr).sort_stats(pstats.SortKey.TIME)
    # stats.print_stats()
    return images


def name_faces(face_name: str,faces_indx: int,not_known_faces: list[tuple[int,Image.Image]]):
    print(len(not_known_faces))
    if faces_indx == 0:
        for i in get_faces_to_name():
            not_known_faces.append(i)
        if len(not_known_faces) > 0:
            return [not_known_faces[0][1]], faces_indx+1
    
    if faces_indx < len(not_known_faces):
        image = not_known_faces[faces_indx][1]
        return [image], faces_indx+1
    else:
        return [], faces_indx

def search_images(search_prompt):
    print(search_prompt)
    return search_prompt



css = """
#upload_button {
    width: 100%; !important;
}
"""

with gr.Blocks(css=css) as demo: # add delete_cache=(86000,86000) to erase the images after 24 hours or after server restart
    faces_index = gr.State( 0)
    not_known_faces = gr.State([])
    with gr.Row():
        gr.Markdown(f"""
            AI Image Search
            """)
    with gr.Row():
        with gr.Column(elem_id="col-container"):

            with gr.Row():
                files = gr.Files(label="Upload Images", file_types=["png", "jpg", "jpeg"])
                
            with gr.Row():
                upload_button = gr.Button("Upload",elem_id="upload_button",size='lg')

            with gr.Row():
                search_prompt = gr.Text(
                    label="Search Prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your search prompt",
                    container=False,
                )
                search_button = gr.Button("Search", scale=0)
            
            with gr.Row():
                name_face_prompt = gr.Text(
                    label="FaceName",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter the face name in the same order as the faces in the gallery",
                    container=False,
                )
                name_face_button = gr.Button("Name Face", scale=0)
        with gr.Column(elem_id="col-container-images"):
            images = gr.Gallery(selected_index=0)
        
    gr.on(
        triggers=[name_face_button.click],
        fn=name_faces,
        inputs=[
            name_face_prompt,
            faces_index,
            not_known_faces
        ],
        outputs=[images,faces_index],
    )

    gr.on(
        triggers=[search_button.click],
        fn=search_images,
        inputs=[
            search_prompt
        ],
        outputs=[images],
    )

    gr.on(
        triggers=[upload_button.click],
        fn=detect_faces,
        inputs=[
            files
        ],
        outputs=[images],
    )

demo.launch()