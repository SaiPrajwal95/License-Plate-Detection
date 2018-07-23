# License-Plate-Detection
This repository hosts the code for license plate detection. CNN based bounding box regression is used to predict the coordinates for the license plate in a given frame. The code is currently under development and isn't complete.

### Requirements
- Python 3.6
- Tensorflow 1.8.0
- Keras 2.2.0

### Steps to run the code
- First, download a dataset of images having vehicles with visible license plates. You could use the following repository for this task: https://github.com/hardikvasa/google-images-download
- After downloading, run the Basic-Annotation-Script.py script to annotate the images in the downloaded dataset. This script finally creates a json file with the annotated data corresponding to all images in the dataset.
- Run train-classifier.py to train the custom CNN to predict the bounding box corresponding to license plates.
- Finally, run test-classifier.py to test the network on new images.

### Sample results
![Sample prediction 1](https://github.com/SaiPrajwal95/License-Plate-Detection/blob/master/Results/Sample_Prediction.png)

![Sample prediction 2](https://github.com/SaiPrajwal95/License-Plate-Detection/blob/master/Results/Sample_Prediction_2.png)
