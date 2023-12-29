import numpy as np
import tensorflow as tf
import cv2
import requests

def read_data (url,shape = (224,224,3)) :
    '''
    Return the np.array of image @url
    '''
    width,height,channels = shape
    response = requests.get(url, stream=True)
# Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Convert the image content to a NumPy array
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)

        # Decode the NumPy array to an OpenCV image
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Resize the image to the desired dimensions
        image = cv2.resize(image, (width, height))

        # Return the preprocessed image
        return image


    return np.zeros((width, height, channels))

def GetTensor(df):
    text_embeds = []
    image_embeds = []
    others = []
    company_embeds = []

    for index, row in df.iterrows():
        text_embeds.append(row['text_embeds'])
        image_embeds.append(row['image_embeds'])
        company_embeds.append(row['company_embedd'])
        others.append(np.array(row[['sentiment','views', 'day_sin', 'day_cos', 'week_sin', 'week_cos', 'year_sin', 'year_cos','uptime', 'mention_count', 'hyperlink_count', 'hashtag_count', 'word_count']]))    
    others = tf.convert_to_tensor(others)
    text_embeds = tf.convert_to_tensor(text_embeds)
    image_embeds = tf.convert_to_tensor(image_embeds)
    company_embeds = tf.convert_to_tensor(company_embeds)
    return [text_embeds, image_embeds, others, company_embeds]


