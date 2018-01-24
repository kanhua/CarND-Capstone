from detect_traffic_light import detect_object_single,\
    load_graph,classify_color,select_boxes,classify_color_v2
import tensorflow as tf
import numpy as np
import cv2
import random
import string
import os

def record_image(cv_image, ref_state, save_path):
    image_saving_frequency = 1.0
    random_val = random.random()
    if random_val < image_saving_frequency:
        random_str = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(8)])

        cv2.imwrite(os.path.join(save_path,random_str + "_" + str(ref_state) + '.PNG'), cv_image)

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier

        self.detection_graph=load_graph()
        self.extract_graph_components()
        self.sess=tf.Session(graph=self.detection_graph)

        # run the first session to "warm up"
        dummy_image=np.zeros((100,100,3))
        self.detect_object(dummy_image)

    def extract_graph_components(self):
        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')


    def detect_object(self,image_np):


        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.

        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        # print("boxes",boxes)
        # print("classes",classes)
        # print("scores",scores)

        sel_boxes = select_boxes(boxes=boxes, classes=classes, scores=scores, target_class=10)

        if len(sel_boxes)==0:
            return None

        sel_box = sel_boxes[0]

        im_height, im_width, _ = image_np.shape
        (left, right, top, bottom) = (sel_box[1] * im_width, sel_box[3] * im_width,
                                      sel_box[0] * im_height, sel_box[2] * im_height)

        cropped_image = image_np[int(top):int(bottom), int(left):int(right), :]

        return cropped_image



    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction

        cropped_image=self.detect_object(image)
        if cropped_image is None:
            return 4
        #cropped_image=image
        #bgr_cropped_image=cv2.cvtColor(cropped_image,cv2.COLOR_RGB2BGR)
        #record_image(bgr_cropped_image,0,"./cropped_data")
        classifed_index,_=classify_color_v2(cropped_image)

        return classifed_index