import gradio as gr
from face_recognition import detect_face
from embedding_calculator import calculate_embedding
from PIL import Image
import numpy as np
import time
import cProfile
import pstats

def detect_faces(files):
    # with cProfile.Profile() as pr:
    images = []
    total_time = 0
    a = time.time()
    for file in files:
        img = Image.open(file)
        new_img,faces = detect_face(np.array(img))
        embeddings = calculate_embedding(faces)
        print(len(embeddings[0]))
        images.append(new_img)
    total_time = time.time() - a
    print(f"Total time taken: {total_time}")
    # stats = pstats.Stats(pr).sort_stats(pstats.SortKey.TIME)
    # stats.print_stats()
    return images


def search_images(search_prompt):
    print(search_prompt)
    return search_prompt



css = """
#upload_button {
    width: 100%; !important;
}
"""

with gr.Blocks(css=css,delete_cache=(80000,70000)) as demo:
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

        with gr.Column(elem_id="col-container-images"):
            images = gr.Gallery(selected_index=0)
        

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