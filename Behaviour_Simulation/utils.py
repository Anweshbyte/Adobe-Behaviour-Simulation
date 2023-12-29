import numpy as np
import json
import os
import requests
import cv2

def read_json_file(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data



def normalize_df(df):
    norms_path = "norms.json"
    norms = read_json_file(norms_path)
    for feature in norms.keys():
        # Convert the list of features to a NumPy array
        features_array = np.array(df[feature])
        max_value = norms[feature]
        df[feature] = features_array/max_value

    return df

def read_data(img_shape, url):
    # Your image reading logic here
    response = requests.get(url)
    if response.status_code == 200:
      img = cv2.imdecode(np.frombuffer(response.content, np.uint8), -1)
      return img
    else:
      return np.zeros(img_shape)
    
def download_images(df):
  output_dir = "images"
  os.makedirs(output_dir, exist_ok=True)
  for index,row in df.iterrows():
    id = row['id']
    link = row['url']
    image_path = output_dir+f'/{id}.jpg'
    if os.path.exists(image_path):
      print(f"Image {id}.jpg already exists. Skipping download.")
      continue
    img = read_data((244,244,3),link)
    if img is not None:
        # Save the image using cv2.imwrite
        print(f"Image {id}.jpg saved successfully")
        cv2.imwrite(image_path, img)
    else:
        print(f"Failed to load image from {link}")