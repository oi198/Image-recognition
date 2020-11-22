from keras.models import load_model
from google.colab import drive
drive.mount('/content/drive')


import numpy as np
from google.colab.patches import cv2_imshow
from PIL import Image

file = '/content/drive/My Drive/ファイル名.PNG'
img = Image.open(file)
img = img.convert('L')
img = img.resize((28, 28)) 
img = np.asarray(img)
img = img.astype('float32')/255
img = np.reshape(img, [28, 28, 1])

model = load_model("/content/drive/My Drive/model.h5")
img = np.array([img])
model.predict_classes(img)
