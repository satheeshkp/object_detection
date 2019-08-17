# Simple object detection example using dlib with a very small dataset

##to run this project

1.first annotate the objects you want to detect using the given command

python gather_annotations.py --dataset clocks --annotations anot.npy --images images.npy

2.train the annotated images using given command

python train.py --annotations anot.npy --images images.npy --detector clock_detector.svm


3.test the object detector using given command

python test.py --detector clock_detector.svm --image test_clock.jpg --annotate Clock

