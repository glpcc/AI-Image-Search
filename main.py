import gradio as gr
from utils.face_recognition import detect_face
from utils.embedding_calculator import calculate_embedding
from utils import dbutils
from PIL import Image
import numpy as np
import time
import cProfile
import pstats

def detect_faces(files: list[str],annotated_images: list[tuple[str,list[tuple[str,str]]]],annotated_images_index: int):
    # with cProfile.Profile() as pr:
    annotated_images.clear()
    total_time = 0
    a = time.time()
    for i,file in enumerate(files):
        img = Image.open(file)
        # Detect the faces in the image
        faces = detect_face(np.array(img))
        # Continue if there are no faces in the image
        if len(faces) == 0:
            continue
        # Calculate the embeddings of the faces
        embeddings = calculate_embedding(faces)
        # Get if there is any similar face in the database
        faces_names_and_ids = dbutils.extract_face_name(embeddings)
        # print(faces_names_and_ids)
        # Store the image in the database
        image_id = dbutils.store_image(file)[0]
        # Store the faces in the database
        faces_names_and_ids = dbutils.store_face(faces,faces_names_and_ids,embeddings,image_id)
        # print(faces_names_and_ids)
        new_annotated_image = (file,[])
        for face,face_name_and_id in zip(faces,faces_names_and_ids):
            x2 = face['facial_area']['x'] + face['facial_area']['w']
            y2 = face['facial_area']['y'] + face['facial_area']['h']
            new_annotated_image[1].append(((face['facial_area']['x'],face['facial_area']['y'],x2,y2),face_name_and_id[1]))
        annotated_images.append(new_annotated_image)
        print(f"Image {i+1} processed of {len(files)}")
    total_time = time.time() - a
    print(f"Total time taken: {total_time}")
    # stats = pstats.Stats(pr).sort_stats(pstats.SortKey.TIME)
    # stats.print_stats()
    return annotated_images[0], 0


def name_faces(face_name: str,faces_indx: int,not_known_faces: list[tuple[int,Image.Image]]):
    if faces_indx == 0:
        temp = dbutils.get_faces_to_name()
        not_known_faces.clear()
        for face in temp:
            not_known_faces.append(face)

        if len(not_known_faces) > 0:
            return [not_known_faces[0][1]], faces_indx+1
        else:
            return [], 0

    if face_name != "":
            dbutils.store_face_name(not_known_faces[faces_indx-1][0],face_name)
    if faces_indx < len(not_known_faces):
        image = not_known_faces[faces_indx][1]
        return [image], faces_indx+1
    else:
        return [], 0

def search_images(search_prompt):
    print(search_prompt)
    return search_prompt


def next_annotated_image(annotated_images: list[tuple[str,list[tuple[str,str]]]],annotated_images_index: int):
    if annotated_images_index < len(annotated_images)-1:
        return annotated_images[annotated_images_index+1], annotated_images_index+1
    else:
        return annotated_images[annotated_images_index], annotated_images_index

def prev_annotated_image(annotated_images: list[tuple[str,list[tuple[str,str]]]],annotated_images_index: int):
    if annotated_images_index > 0:
        return annotated_images[annotated_images_index-1], annotated_images_index-1
    else:
        return annotated_images[annotated_images_index], annotated_images_index

css = """
#upload_button {
    width: 100%; !important;
}
"""

with gr.Blocks(css=css,delete_cache=(86000,86000)) as demo: # add delete_cache=(86000,86000) to erase the images after 24 hours or after server restart
    faces_index = gr.State( 0)
    not_known_faces = gr.State([])
    annotated_images = gr.State([])
    annotated_images_index = gr.State(0)
    with gr.Row():
        gr.Markdown(f"""
            AI Image Search
            """)
    with gr.Row():
        with gr.Column(elem_id="col-container"):

            with gr.Row():
                files = gr.Files(label="Upload Images", file_types=["png", "jpg", "jpeg"],height=250)
                
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
            with gr.Row():
                clean_db_button = gr.Button("Clean Database", scale=0)
        with gr.Column(elem_id="col-container-images"):
            with gr.Row():
                images = gr.Gallery(selected_index=0)
            with gr.Row():
                annotated_image = gr.AnnotatedImage()
            with gr.Row():
                prev_button = gr.Button("Prev")
                next_button = gr.Button("Next")
                
    gr.on(
        triggers=[next_button.click],
        fn=next_annotated_image,
        inputs=[annotated_images,annotated_images_index],
        outputs=[annotated_image,annotated_images_index],
    )
    gr.on(
        triggers=[prev_button.click],
        fn=prev_annotated_image,
        inputs=[annotated_images,annotated_images_index],
        outputs=[annotated_image,annotated_images_index],
    )
    gr.on(
        triggers=[clean_db_button.click],
        fn=dbutils.clean_db,
        inputs=[],
        outputs=[],
    )
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
            files,
            annotated_images,
            annotated_images_index
        ],
        outputs=[annotated_image,annotated_images_index],
    )

demo.launch()