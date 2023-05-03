from tensorflow.keras.models import load_model
from urllib.request import urlopen
import base64
import numpy as np
import cv2

# Load model
reconstructed_janus = load_model("janus")

# Load image as data URI
image_path = "final-dataset/train/weighted/AFPPopup.png"
binary_data = open(image_path, 'rb').read()
image_enc = base64.b64encode(binary_data)
image_ext = image_path.split(".")[-1]
image_uri = "data:image/" + image_ext + ";base64," + image_enc.decode("utf-8")
print(image_uri)

# Read image from data URI
response = urlopen(image_uri)
image_data = response.read()
img_arr = np.asarray(bytearray(image_data), dtype=np.uint8)
img_arr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
img_arr=cv2.resize(img_arr,(224,224))
test_input=np.array([img_arr])
test_input=test_input/225.0

# Let's check:
print(reconstructed_janus.predict(test_input).argmax(axis=1)[0])

# 0 is absent, 1 is even, 2 is weighted
