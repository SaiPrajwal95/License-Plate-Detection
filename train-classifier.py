import json
import cv2
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.layers import Dropout
from keras.models import load_model
from matplotlib import pyplot as plt
from IPython.display import clear_output

# Manual image data generation
def ManualImageDataGenerator(annotation_data_dict, test_split_ratio):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    test_indeces = list(np.random.randint(0,len(annotation_data_dict),int(test_split_ratio*len(annotation_data_dict))))
    indx = 0
    for ann in annotation_data_dict:
        img = cv2.imread(ann['filepath'],1)
        x1 = ann['bboxes'][0][0] * (128/img.shape[1])
        y1 = ann['bboxes'][0][1] * (64/img.shape[0])
        x2 = ann['bboxes'][0][2] * (128/img.shape[1])
        y2 = ann['bboxes'][0][3] * (64/img.shape[0])
        ann['bboxes'][0] = [x1, y1, x2, y2]
        img = cv2.resize(img,(128,64))
        if indx in test_indeces:
            x_test.append(img)
            y_test.append(ann['bboxes'][0])
        else:
            x_train.append(img)
            y_train.append(ann['bboxes'][0])
        indx += 1
    final_train_data = (np.array(x_train),np.array(y_train))
    final_test_data = (np.array(x_test),np.array(y_test))
    return final_train_data, final_test_data

# Initialising the CNN
classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 128, 3), kernel_initializer='he_normal'))
classifier.add(BatchNormalization())
classifier.add(Activation('elu'))
# classifier.add(Dropout(0.5))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(64, (3, 3), kernel_initializer='he_normal'))
classifier.add(BatchNormalization())
classifier.add(Activation('elu'))
# classifier.add(Dropout(0.5))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(128, (3, 3), kernel_initializer='he_normal'))
classifier.add(BatchNormalization())
classifier.add(Activation('elu'))
# classifier.add(Dropout(0.5))

classifier.add(Flatten())

classifier.add(Dense(units = 500, activation = 'elu', kernel_initializer='he_normal'))
classifier.add(Dense(units = 500, activation = 'elu', kernel_initializer='he_normal'))
classifier.add(Dense(units = 4, activation = 'relu', kernel_initializer='he_normal'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mae'])

with open('annotation_data.json','r') as f:
    data = json.load(f)

raw_training_data, raw_testing_data = ManualImageDataGenerator(data,0.1)
(x_train, y_train), (x_test, y_test) = raw_training_data, raw_testing_data

from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=False,
    rotation_range=0,
    zca_whitening = False,
    width_shift_range=0,
    height_shift_range=0,
    horizontal_flip=False)

datagen.fit(x_train)
classifier.fit_generator(datagen.flow(x_train, y_train, batch_size=8),
                    steps_per_epoch=2*len(x_train) / 8, epochs=20)

classifier.save('trained_model.h5')
