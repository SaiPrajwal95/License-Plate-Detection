import cv2
import json
import numpy as np
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

import h5py
from keras.models import load_model
classifier = load_model('trained_model.h5')

with open('annotation_data.json','r') as fp:
    data = json.load(fp)

indx = np.random.randint(0,len(data))
img_path = data[indx]['filepath']
# img_path = '/home/saiprajwalk/Desktop/test_5.jpg'
IMG = cv2.imread(img_path,1)
IMG = cv2.resize(IMG,(128,64))
IMG = np.expand_dims(IMG, axis=0)
IMG.shape
preds = classifier.predict(IMG)
print(preds)

IMG = cv2.imread(img_path,1)
r, c, _ = IMG.shape
IMG = cv2.rectangle(IMG,(int(preds[0][0]*c/128),int(preds[0][1]*r/64)),(int(preds[0][2]*c/128),int(preds[0][3]*r/64)),(0,0,255),2)
cv2.imshow("pred", IMG)
cv2.waitKey(0)
cv2.destroyAllWindows()
