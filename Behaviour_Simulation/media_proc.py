import requests
from PIL import Image
import requests
import cv2
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from io import BytesIO

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def process_image_with_text(image_url,sentence):
    # Image processing logic
    response = requests.get(image_url, stream=True).raw
    image = Image.open(response)

    # Resize the image using OpenCV
    image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # print(image_array.shape)
    resized_image = cv2.resize(image_array, (244, int((image_array.shape[0] / image_array.shape[1]) * 244)))
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    # print(resized_image.shape)
    resized_image = Image.fromarray(resized_image)
    
    # Process the resized image
    inputs = processor(text=[sentence], images=resized_image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    text_embeds = outputs['text_embeds']
    image_embeds = outputs['image_embeds']

    return text_embeds.tolist(),image_embeds.tolist()

def process_whiteimage_with_text():
    # white image to replace corrupt image
    image_url='https://pbs.twimg.com/media/Eo8N3JLVoAAlDJT?format=jpg&name=small'
    sentence='White Image'
    return process_image_with_text(image_url,sentence)

def process_video_with_text(video_url,thumbnail_url,sentence):
    # Video processing logic
    cap = cv2.VideoCapture(video_url)

    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize variables
    prev_frame = None
    key_frames = []
    frame_list = []

    ret, first_frame = cap.read()
    # if not ret:
    #     return key_frames, fps, frame_list  # No frames in the video

    # Convert the first frame to grayscale
    gray_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Add the first frame to the frame_list
    frame_list.append(Image.fromarray(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)))

    j = 0
    while j<4000:
        ret, frame = cap.read()
        j += 1
        if not ret:
            break   

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate the absolute difference between the current and previous frames
    if prev_frame is not None:
        diff = cv2.absdiff(prev_frame, gray_frame)

        # Calculate the percentage of pixels that are different
        percentage_diff = np.sum(diff) / float(gray_frame.size)

        # If the percentage difference is above the threshold, consider it a key frame
        if percentage_diff > 50:
            key_frames.append((int(cap.get(cv2.CAP_PROP_POS_FRAMES)), frame))
            frame_list.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    prev_frame = gray_frame

    # Release the video capture object
    cap.release()

    stacked_frames = []

    # Iterate through each frame in the list
    for frame in frame_list:
        # Append the frame to the stacked_frames list
        stacked_frames.append(frame)

    # Stack the frames along the 0-axis, calculate the mean, and convert to uint8
    stacked_image = np.stack(stacked_frames, axis=0).mean(axis=0).astype(np.uint8)
    stacked_image_pil = Image.fromarray(stacked_image)

    if len(frame_list) == 1:
        response = requests.get(thumbnail_url)
        img = Image.open(BytesIO(response.content))
        frame_list.append(img)

    # Process the stacked image
    inputs = processor(text=[sentence], images=stacked_image_pil, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    text_embeds = outputs['text_embeds']
    image_embeds = outputs['image_embeds']

    return text_embeds.tolist(),image_embeds.tolist()


# Assuming you have the processor and model initialized

def embed_df(df): 
# def embed_df(df,name): #
    text_embeds_batch = []
    image_embeds_batch = []
    # urls = df['url'].tolist()
    # sentences = df['content'].tolist()
    for index,row in df.iterrows():
        url = row['url']
        sentence = row['content']
        id = row['id']
        try:
            image_path = f"images/{id}.jpg"
            # image_path = f"images/{name}/{id}.jpg" #
            # response = requests.get(url, stream=True).raw
            # image_data = BytesIO(response.content)
            # image = Image.open(image_data)
            image = Image.open(image_path)

            # Resize the image using OpenCV
            image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            resized_image = cv2.resize(image_array, (244, int((image_array.shape[0] / image_array.shape[1]) * 244)))
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            resized_image = Image.fromarray(resized_image)

            inputs = processor(text=[sentence], images=resized_image, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            text_embeds = outputs['text_embeds']
            image_embeds = outputs['image_embeds']

            text_embeds_batch.append(text_embeds.tolist())
            image_embeds_batch.append(image_embeds.tolist())

        except Exception as e:
            print(f"Error processing {url}: {e}")
            text_embed,white_image_embed = process_whiteimage_with_text()
            text_embeds_batch.append(text_embed)
            image_embeds_batch.append(white_image_embed)


    df['text_embeds'] = text_embeds_batch
    df['image_embeds'] = image_embeds_batch

    return df

