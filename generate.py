from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

# Reconstructing janus:
reconstructed_janus = load_model("janus")

# Declaring the path to the image:
image_path = "final-dataset/train/weighted/AFPPopup.png"

# Transforming the image into a standardized numpy array:
img_arr=cv2.imread(image_path)
img_arr=cv2.resize(img_arr,(224,224))
test_input=np.array([img_arr])
test_input=test_input/225.0

# Checking and printing the result:
# 0 is absent, 1 is even, 2 is weighted
print(reconstructed_janus.predict(test_input).argmax(axis=1)[0])