#usage python train.py --annotations anot.npy --images images.npy --detector clock_detector.svm

import numpy as np
import argparse


import dlib
import cv2


class ObjectDetector(object):
    def __init__(self,options=None,loadPath=None):
        #create detector options
        self.options = options
        if self.options is None:
            self.options = dlib.simple_object_detector_training_options()

        #load the trained detector (for testing)
        if loadPath is not None:
            self._detector = dlib.simple_object_detector(loadPath)

    def _prepare_annotations(self,annotations):
        annots = []
        for (x,y,xb,yb) in annotations:
            annots.append([dlib.rectangle(left=int(x),top=int(y),right=int(xb),bottom=int(yb))])
        return annots

    def _prepare_images(self,imagePaths):
        images = []
        for imPath in imagePaths:
            image = cv2.imread(imPath)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            images.append(image)
        return images

    def fit(self, imagePaths, annotations, visualize=False, savePath=None):
        annotations = self._prepare_annotations(annotations)
        images = self._prepare_images(imagePaths)
        self._detector = dlib.train_simple_object_detector(images, annotations, self.options)

        #visualize HOG
        if visualize:
            win = dlib.image_window()
            win.set_image(self._detector)
            dlib.hit_enter_to_continue()

        #save detector to disk
        if savePath is not None:
            self._detector.save(savePath)

        return self

    def predict(self,image):
        boxes = self._detector(image)
        preds = []
        for box in boxes:
            (x,y,xb,yb) = [box.left(),box.top(),box.right(),box.bottom()]
            preds.append((x,y,xb,yb))
        return preds

    def detect(self,image,annotate=None):
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        preds = self.predict(image)
        for (x,y,xb,yb) in preds:
            image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

            #draw and annotate on image
            cv2.rectangle(image,(x,y),(xb,yb),(0,0,255),2)
            if annotate is not None and type(annotate)==str:
                cv2.putText(image,annotate,(x+5,y-5),cv2.FONT_HERSHEY_SIMPLEX,1.0,(128,255,0),2)
        cv2.imshow("Detected",image)
        cv2.waitKey(0)


ap = argparse.ArgumentParser()
ap.add_argument("-a","--annotations",required=True,help="path to saved annotations...")
ap.add_argument("-i","--images",required=True,help="path to saved image paths...")
ap.add_argument("-d","--detector",default=None,help="path to save the trained detector...")
args = vars(ap.parse_args())

print ("[INFO] loading annotations and images")
annots = np.load(args["annotations"])
imagePaths = np.load(args["images"])

detector = ObjectDetector()
print ("[INFO] creating & saving object detector")

detector.fit(imagePaths,annots,visualize=True,savePath=args["detector"])
