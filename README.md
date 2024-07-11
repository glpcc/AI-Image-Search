# AI-Image-Search
A personal project to develop a intelligent Image search engine using face recognition and embedding comparison. The GUI is made with [Gradio](https://www.gradio.app/), the choosen database is [Clickhouse](https://clickhouse.com/) ( A bit overkill ), the face detection and recognition is made using the [Deepface Module](https://github.com/serengil/deepface) and the embedding calculations from images and text are made with the [Transformers Module](https://github.com/huggingface/transformers) using the [Siglip Vision and Text Models](https://huggingface.co/docs/transformers/main/model_doc/siglip).

# Installation 
## Disclaimer
All test were made on my machine locally with an RTX4070 Nvidia GPU with 12Gb of VRAM so it is highly recommended to have at least the same amount of VRAM in your system.

If you want to run this on a lower end system you might want to change the face recognition and face detection models in the [MLutils.py](/utils/MLutils.py) file (change "retinaface" and "facenet512" to lower end ones like "opencv" and "VGG-Face" respectively, the result quality will be hindered)

## Pre-needed software
You will need python installed (preferably >=3.10) and [Docker](https://www.docker.com/) 

## Dependecies
To install all dependencies run the next command in the project folder **the torch version asummes you have a cuda enabled GPU!!**
```powershell
pip install -r requirements.txt
```

## Running the interface
### Clickhouse Password
Before you run anything, especially if you are going to publish it online, change the CLICKHOUSE_PASSWORD in the [.env](.env) file

Before running the project run the docker compose in a terminal in the project folder
```bash
docker compose up
```
Then, run the main.py file and after a bit a localhost link will be printed to the interface web.
```bash
python main.py
```

# Usage 
## Upload images 
To upload some images to the database, just click on the *upload here* area and select one or more images to be uploaded, the press the upload button and the processing will start. When the processing step is done (around 3s for the first image and 1s per image afterward depending on the image size) you will see something like in the next image, with a color rectangle around the faces and the person name below (Not named will appear if no prevously named face matches).

![](/imgs/Recognition.png)
Press the next and previous buttons to navigate through all the uploaded images

## Name Faces
To name the stored faces, first select the number of times you want the face to have appeared in the uploaded images to be worth naming. Then press the Name Face button and a croped image of the face will appear on the right, write the name on the text box and press again the name face button. Repeat until no more faces appear.

## Search By Text
To search the images by content using natural language search, first select the maximun number of images you want to see, then just enter the prompt on the textbox on the left of the search button and press it. After this, a gallery with the images will appear on the right sorted left to right, from top to bottom by most similarity to the prompt.

Here are some examples:

![](/imgs/Search_by_text1.png)
![](/imgs/Search_by_text2.png)
Disclaimer: if no similar images were uploaded the results might be non sensical.

## Search By Face
First, if you just named some faces in the same session press the *update face names* button to see the checkboxes. Then select one or more names you want to appear in the images, the result will be only images in which all the persons appear.

Here are some examples:

![](/imgs/Search_by_face1.png)
![](/imgs/Search_by_face2.png)

# Posible future optimizations 
- Posible bathching of the face recognition step
- Image resolution downsampling on upload for faster inference
- Distribution of models to make some parts async, like the image embedding calculation
- Containerize the models to run on nvidia nims.