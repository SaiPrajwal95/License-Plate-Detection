import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import json

drawing = False # true if mouse is pressed
ix,iy = -1,-1

# mouse callback function
def draw_annotation(event,x,y,flags,param):
    global ix,iy,drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),2)
        print(ix,iy,x,y)
        img_data['bboxes'].append((ix,iy,x,y))

if __name__ == '__main__':
    path = input('Enter the path to images directory: ')
    data_annotations = []
    for f in listdir(path):
        if isfile(join(path, f)):
            img_data = {}
            img_data['filepath'] = path+'/'+f
            img_data['bboxes'] = []
            # Image read and annotate here
            img = cv2.imread(img_data['filepath'],1)
            try:
                x, y, c = img.shape
            except:
                continue
            img_data['size'] = (x,y,c)
            cv2.namedWindow('image',cv2.WINDOW_NORMAL)
            cv2.setMouseCallback('image',draw_annotation)
            check = 0
            while(1):
                cv2.imshow('image',img)
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    check = 1
                    break
                elif k == ord('q'):
                    break
            # End of draw_annotation
            cv2.destroyAllWindows()
            if(len(img_data['bboxes'])>0):
                data_annotations.append(img_data)
            if(check):
                break
    with open('annotation_data.json','w') as f:
        json.dump(data_annotations, f, indent=2)
