import fastdup
import cv2
import os
import numpy as np
from utils import read_data,extract_id

def download_images(df):
    os.mkdir("images")
    for index,row in df.iterrows():
        id = row['id']
        url = row['url']

        img = read_data(url)
        cv2.imwrite(f"images/{id}.jpg",img)

    
def get_captions(num_images,device = "gpu",images_dir = "images"):

    fd = fastdup.create(input_dir=images_dir)
    fd.run(ccthreshold=0.9, num_images= num_images, overwrite=True)
    captions_df = fd.caption(model_name='blip', device = device, batch_size=256)
    captions_df['id'] = captions_df['filename'].apply(extract_id)

    captions_df.set_index('id', inplace=True)

    captions = []
    for id in range(num_images):
        captions.append(captions_df[id])

    return captions