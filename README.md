This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).


## Team Lead and Member

Kan-Hua Lee/ kanhwa@gmail.com (single member)


## Installation and Requirements

Detailed installation instruction to run this project is described [here](./install.md)

This project requires

- Python 2.7.12
- Tensorflow 1.3.0
- OpenCV 3.3.1


## Notes on Traffic light Classification Model

My classification model requires a trained MobileNet model.
It will be downloaded automatically when first running this project.

If somehow the model cannot be downloaded automatically, you can download it from [this link](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz) and
extract it to folder ```tl_detector/```.

Since my VM does not have a GPU to run Tensorflow,
I need to reduce the number of images to be classified to not to drag the car.
The default setting only run 1/20 of the images sending to the ```tl_detector```.
This can be altered by changing ```DETECT_FREQ``` in ```tl_detector.py```.