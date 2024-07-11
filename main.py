import gradio as gr
import utils.MLutils as MLutils
from utils import dbutils
from PIL import Image
import numpy as np
import time
import random
import cProfile
import pstats

def proccess_images(files: list[str],annotated_images: list[tuple[str,list[tuple[str,str]]]],annotated_images_index: int):
    # with cProfile.Profile() as pr:
    annotated_images.clear()
    total_time = 0
    a = time.time()
    images_ids = []
    for i,file in enumerate(files):
        img = Image.open(file)
        # Detect the faces in the image
        faces = MLutils.detect_face(np.array(img))

        # Generate a random image id
        image_id = random.randint(0,2**64)
        images_ids.append(image_id)

        # Continue if there are no faces in the image
        if len(faces) == 0:
            print(f"Image {i+1} processed of {len(files)} with no faces detected")
            annotated_images.append((file,[]))
            continue
        # Calculate the embeddings of the faces
        embeddings = MLutils.calculate_face_embedding(faces)

        # Get if there is any similar face in the database
        faces_names_and_ids = dbutils.extract_face_name(embeddings)
        # print(faces_names_and_ids)

        
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

    #calculate the embeddings of the images and store them in the database
    store_images(files,images_ids)
    print("Images stored in the database")
    # stats = pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE)
    # stats.print_stats(50)
    if len(annotated_images) > 0:
        return annotated_images[0], 0
    else:
        raise Exception("No images to show")

def store_images(files: list[str], images_ids: list[int]):
    embeddings = MLutils.calculate_images_embedding(files)
    dbutils.store_images(files,images_ids,embeddings)

def name_faces(face_name: str,faces_indx: int,not_known_faces: list[tuple[int,Image.Image]],num_of_face_appeareces_slider_value: float):
    num_of_face_appeareces_slider_value = int(num_of_face_appeareces_slider_value)
    if faces_indx == 0:
        temp = dbutils.get_faces_to_name(num_of_face_appeareces_slider_value)
        not_known_faces.clear()
        for face in temp:
            not_known_faces.append(face)

        if len(not_known_faces) > 0:
            return [not_known_faces[0][1]], faces_indx+1
        else:
            return [], 0

    if face_name != "":
        face_name = face_name.strip().lower()
        dbutils.store_face_name(not_known_faces[faces_indx-1][0],face_name)
    if faces_indx < len(not_known_faces):
        image = not_known_faces[faces_indx][1]
        return [image], faces_indx+1
    else:
        return [], 0

def search_images(search_prompt: str,max_images: float):
    max_images = int(max_images)
    text_embedding = MLutils.calculate_text_embedding(search_prompt)
    image_list = dbutils.search_images_by_embbeding(text_embedding,max_images)
    image_list = [(image,f'Score {200 - score*100:.2f}') for image,score in image_list]
    return image_list

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

def search_by_face(multi_select_faces: list[str],posible_choises_state: dict[str,list[int]]):
    faces_ids = []
    num_repeated_face_names = 0
    for face_name in multi_select_faces:
        faces_ids.extend(posible_choises_state[face_name])
        num_repeated_face_names += len(posible_choises_state[face_name]) - 1
    image_list = dbutils.get_images_by_face(faces_ids,num_repeated_face_names)
    image_result = [(image,f'Image {i}') for i,image in enumerate(image_list)]
    return image_result

def update_face_names():
    posible_choises = dbutils.get_named_faces()
    return posible_choises, gr.CheckboxGroup(choices=list(posible_choises.keys()),container=False)

css = """
#upload_button {
    width: 100%; !important;
}
"""
def visibility_logic(func = None, *args, **kwargs):
    if func == name_faces or func == search_by_face or func == search_images:
        return (gr.Row(visible=True),gr.Row(visible=False),gr.Row(visible=False))
    elif func == proccess_images:
        return (gr.Row(visible=False),gr.Row(visible=True),gr.Row(visible=True))
    


with gr.Blocks(css=css) as demo: # add delete_cache=(86000,86000) to erase the images after 24 hours or after server restart
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

            with gr.Row(visible=True):
                search_prompt = gr.Text(
                    label="Search Prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your search prompt",
                    container=False,
                )
                search_button = gr.Button("Search", scale=0)

            with gr.Row(visible=True):
                max_images = gr.Slider(label="Max Images", minimum=1, maximum=100, step=1, value=30)
            
            with gr.Row():
                name_face_prompt = gr.Text(
                    label="FaceName",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter the face name of the person in the image, leave empty if you don't know the person",
                    container=False,
                )
                name_face_button = gr.Button("Name Face", scale=0)
            with gr.Row(visible=True):
                num_of_face_appeareces_slider = gr.Slider(label="Number of times the face must appear to name it", minimum=1, maximum=10, step=1, value=2)
            with gr.Row():
                posible_choises = dbutils.get_named_faces()
                posible_choises_state = gr.State(posible_choises)
                multi_select_faces = gr.CheckboxGroup(
                    label="Select the faces you want to appear in the gallery",
                    choices=list(posible_choises.keys()),
                    container=False,
                )
                search_by_face_button = gr.Button("Search by Face", scale=0)
            with gr.Row():
                clean_db_button = gr.Button("Clean Database", scale=0)
                update_face_names_button = gr.Button("Update face names", scale=0)
        with gr.Column(elem_id="col-container-images"):
            with gr.Row(visible=False) as images_container:
                images = gr.Gallery(selected_index=0)
            with gr.Row(visible=False) as annotated_image_container:
                annotated_image = gr.AnnotatedImage()
            with gr.Row(visible=False) as buttons_container:
                prev_button = gr.Button("Prev")
                next_button = gr.Button("Next")
    # Set the triggers

    # Trigger for the next annotated image          
    gr.on(
        triggers=[next_button.click],
        fn=next_annotated_image,
        inputs=[annotated_images,annotated_images_index],
        outputs=[annotated_image,annotated_images_index],
    )
    # Trigger for the previous annotated image
    gr.on(
        triggers=[prev_button.click],
        fn=prev_annotated_image,
        inputs=[annotated_images,annotated_images_index],
        outputs=[annotated_image,annotated_images_index],
    )
    # Trigger for the clean database button
    gr.on(
        triggers=[clean_db_button.click],
        fn=dbutils.clean_db,
        inputs=[],
        outputs=[],
    )

    # Trigger to handle the visibility of components to start naming the faces
    gr.on(
        triggers=[name_face_button.click],
        fn=lambda *args,**kwargs: visibility_logic(name_faces,*args,**kwargs),
        inputs=[
        ],
        outputs=[images_container,annotated_image_container,buttons_container],
    )

    # Trigger to start naming the faces
    gr.on(
        triggers=[name_face_button.click],
        fn=name_faces,
        inputs=[
            name_face_prompt,
            faces_index,
            not_known_faces,
            num_of_face_appeareces_slider
        ],
        outputs=[images,faces_index],
    )

    # Trigger for the visibility of the components to start the face detection
    gr.on(
        triggers=[search_button.click],
        fn=lambda *args,**kwargs: visibility_logic(search_images,*args,**kwargs),
        inputs=[],
        outputs=[images_container,annotated_image_container,buttons_container],
    ) 

    # Trigger to search by menaing
    gr.on(
        triggers=[search_button.click],
        fn=search_images,
        inputs=[
            search_prompt,
            max_images
        ],
        outputs=[images],
    )

    # Trigger for the visibility of the components to start the face detection
    gr.on(
        triggers=[upload_button.click],
        fn=lambda *args,**kwargs: visibility_logic(proccess_images,*args,**kwargs),
        inputs=[],
        outputs=[images_container,annotated_image_container,buttons_container],
    ) 

    # Trigger for the face detection
    gr.on(
        triggers=[upload_button.click],
        fn=proccess_images,
        inputs=[
            files,
            annotated_images,
            annotated_images_index
        ],
        outputs=[annotated_image,annotated_images_index],
    )

    # Trigger for the visibility of the components to search by face
    gr.on(
        triggers=[search_by_face_button.click],
        fn=lambda *args,**kwargs: visibility_logic(search_by_face,*args,**kwargs),
        inputs=[],
        outputs=[images_container,annotated_image_container,buttons_container],
    )

    # Trigger for the search by face
    gr.on(
        triggers=[search_by_face_button.click],
        fn=search_by_face,
        inputs=[
            multi_select_faces,
            posible_choises_state
        ],
        outputs=[images],
    )

    # Trigger for the update face names button
    gr.on(
        triggers=[update_face_names_button.click],
        fn=update_face_names,
        inputs=[],
        outputs=[posible_choises_state,multi_select_faces],
    )

demo.launch(share=False)