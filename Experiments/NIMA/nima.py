#A model for neural image assessment 
import numpy as np                                                                                                                                    
from tensorflow.keras.applications.mobilenet import MobileNet 
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.models import Model


weights_path = "Experiments\NIMA\mobilenet_weights.h5"
base_nima_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
x = Dropout(0.75)(base_nima_model.output)
x = Dense(10, activation='softmax')(x)

nima_model = Model(base_nima_model.input, x)
nima_model.load_weights(weights_path)


def get_score(img):
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    scores = nima_model.predict(x, batch_size=1, verbose=0)

    return np.array(scores[0])