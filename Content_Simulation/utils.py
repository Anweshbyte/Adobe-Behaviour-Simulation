import re
import numpy as np
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


def extract_id(input_string):
    # Use regular expression to find the numeric part before the file extension
    match = re.search(r'/(\d+)\.jpg$', input_string)
    
    # Check if a match is found
    if match:
        # Extract the ID from the matched group
        id_number = int(match.group(1))
        return id_number
    else:
        # Return an appropriate value or raise an exception if no match is found
        return None
    

def get_prompt(row):
    date,time,username,inferred_company,likes,caption = row[['date','time','username','inferred company','likes','caption']]

    prompt = f"""
    Generate a tweet that might have been posted:
    📅 Date: {date} , Time: {time}
    👤 Username: {username}
    🏢 Company: {inferred_company}
    👍 Likes: {likes}
    🖼 Media: {caption}
    """
    return prompt
    
