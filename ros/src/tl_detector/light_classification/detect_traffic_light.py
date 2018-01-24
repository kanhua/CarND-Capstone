from __future__ import division
import numpy as np
import os
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import cv2

if tf.__version__ != '1.3.0':
    raise ImportError('Please use tensorflow version 1.3.0')

# What model to download.
# MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
model_path = "./"
PATH_TO_CKPT = model_path + MODEL_NAME + '/frozen_inference_graph.pb'


def load_graph():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    return detection_graph


# select only the classes of traffic light
def select_boxes(boxes, classes, scores, target_class=10):
    """

    :param boxes:
    :param classes:
    :param scores:
    :param target_class: default traffic light id in COCO dataset is 10
    :return:
    """

    sq_scores = np.squeeze(scores)
    sq_classes = np.squeeze(classes)
    sq_boxes = np.squeeze(boxes)

    sel_id = np.logical_and(sq_classes == target_class, sq_scores > 0)

    return sq_boxes[sel_id]


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def detect_object(detection_graph, TEST_IMAGE_PATHS):
    cropped_images = []

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            for image_path in TEST_IMAGE_PATHS:
                image = Image.open(image_path)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.

                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # print("boxes",boxes)
                # print("classes",classes)
                # print("scores",scores)

                sel_boxes = select_boxes(boxes=boxes, classes=classes, scores=scores, target_class=10)
                sel_box = sel_boxes[0]

                im_height, im_width, _ = image_np.shape
                (left, right, top, bottom) = (sel_box[1] * im_width, sel_box[3] * im_width,
                                              sel_box[0] * im_height, sel_box[2] * im_height)

                cropped_image = image_np[int(top):int(bottom), int(left):int(right), :]
                cropped_images.append(cropped_image)
    return cropped_images


def detect_object_single(detection_graph, image_np):
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.

            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # print("boxes",boxes)
            # print("classes",classes)
            # print("scores",scores)

            sel_boxes = select_boxes(boxes=boxes, classes=classes, scores=scores, target_class=10)
            sel_box = sel_boxes[0]

            im_height, im_width, _ = image_np.shape
            (left, right, top, bottom) = (sel_box[1] * im_width, sel_box[3] * im_width,
                                          sel_box[0] * im_height, sel_box[2] * im_height)

            cropped_image = image_np[int(top):int(bottom), int(left):int(right), :]
    return cropped_image


def get_h_image(rgb_image):
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    return hsv_image[:, :, 0]


def high_value_region_mask(hsv_image, v_thres=0.6):
    idx = (hsv_image[:, :, 2].astype(np.float) / 255.0) < v_thres
    mask = np.ones_like(hsv_image[:, :, 2])
    mask[idx] = 0
    return mask


def high_saturation_region_mask(hsv_image, s_thres=0.6):
    idx = (hsv_image[:, :, 1].astype(np.float) / 255.0) < s_thres
    mask = np.ones_like(hsv_image[:, :, 1])
    mask[idx] = 0
    return mask


def channel_percentile(single_chan_image, percentile):
    sq_image = np.squeeze(single_chan_image)
    assert len(sq_image.shape) < 3

    thres_value = np.percentile(sq_image.ravel(), percentile)

    return float(thres_value) / 255.0


def low_saturation_region_mask(hsv_image, s_thres=0.6):
    idx = (hsv_image[:, :, 1].astype(np.float) / 255.0) > s_thres
    mask = np.zeros_like(hsv_image[:, :, 1])
    mask[idx] = 1
    return mask


def low_value_region_mask(hsv_image, v_thres=0.6):
    idx = (hsv_image[:, :, 2].astype(np.float) / 255.0) > v_thres
    mask = np.zeros_like(hsv_image[:, :, 2])
    mask[idx] = 1
    return mask


def get_masked_hue_values(rgb_image):
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    sat_mask = high_saturation_region_mask(hsv_image)
    val_mask = high_value_region_mask(hsv_image)
    masked_hue_image = hsv_image[:, :, 0]
    masked_hue_1d = masked_hue_image[np.logical_and(val_mask, sat_mask)].ravel()
    # scale it from 0-179 to 2pi
    mean_angle = get_mean_hue_value(masked_hue_1d)

    # return the value in [-pi, pi] range
    return mean_angle


def get_masked_hue_values_v2(rgb_image):
    """
    Get the pixels in the RGB image that has high saturation (S) and value (V) in HSV chanels

    :param rgb_image: image (height, width, channel)
    :return: a 1-d array
    """

    hsv_test_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    s_thres_val = channel_percentile(hsv_test_image[:, :, 1], percentile=70)
    v_thres_val = channel_percentile(hsv_test_image[:, :, 2], percentile=70)
    val_mask = high_value_region_mask(hsv_test_image, v_thres=v_thres_val)
    sat_mask = low_saturation_region_mask(hsv_test_image, s_thres=s_thres_val)
    masked_hue_image = hsv_test_image[:, :, 0]
    masked_hue_1d = masked_hue_image[np.logical_and(val_mask, sat_mask)].ravel()

    return masked_hue_1d


def convert_to_hue_angle(hue_array):
    """
    Convert the hue values from [0,179] to radian degrees [-pi, pi]

    :param hue_array: array-like, the hue values in degree [0,179]
    :return: the angles of hue values in radians [-pi, pi]
    """

    hue_cos = np.cos(hue_array * np.pi / 90)
    hue_sine = np.sin(hue_array * np.pi / 90)

    hue_angle = np.arctan2(hue_sine, hue_cos)

    return hue_angle


def get_mean_hue_value(masked_hue_1d):
    masked_hue_1d = masked_hue_1d * np.pi / 90
    # hue values cannot be compared directly. Need to convert it to sine and cosine.
    # an alternative is using complex space exp(i*theta)
    masked_hue_1d_cos = np.mean(np.cos(masked_hue_1d))
    masked_hue_1d_sin = np.mean(np.sin(masked_hue_1d))
    mean_angle = np.arctan2(masked_hue_1d_sin, masked_hue_1d_cos)
    return mean_angle


def classify_color(rgb_image):
    hue_value = get_masked_hue_values(rgb_image)
    # use additional '_' so that the indexes can match
    color_text = ['red', 'yellow', 'green', '_', 'unknown']
    color_hue = np.array([0, 0.333 * np.pi, 0.66 * np.pi])

    value_diff = np.abs(color_hue - hue_value)
    min_index = np.argmin(value_diff)
    if (value_diff[min_index] > 0.33 * np.pi):
        min_index = 4

    return min_index, color_text[min_index]


def classify_color_by_range(hue_value):
    """
    Determine the color (red, yellow or green) in a hue value array

    :param hue_value: hue_value is radians
    :return: the color index ['red', 'yellow', 'green', '_', 'unknown']
    """
    red_index = np.logical_and(hue_value < (0.125 * np.pi), hue_value > (-0.125 * np.pi))

    green_index = np.logical_and(hue_value > (0.66 * np.pi), hue_value < np.pi)

    yellow_index = np.logical_and(hue_value > (0.25 * np.pi), hue_value < (5.0 / 12.0 * np.pi))

    color_counts = np.array([np.sum(red_index) / len(hue_value),
                             np.sum(yellow_index) / len(hue_value),
                             np.sum(green_index) / len(hue_value)])

    color_text = ['red', 'yellow', 'green', '_', 'unknown']

    min_index = np.argmax(color_counts)
    print("color counts:", color_counts)

    return min_index, color_text[min_index]


def classify_color_v2(rgb_image):
    """
    Full pipeline of classifying the traffic light color from the traffic light image

    :param rgb_image: the RGB image array (height,width, RGB channel)
    :return: the color index ['red', 'yellow', 'green', '_', 'unknown']
    """

    hue_1d_deg = get_masked_hue_values_v2(rgb_image)

    hue_1d_rad = convert_to_hue_angle(hue_1d_deg)

    return classify_color_by_range(hue_1d_rad)


if __name__ == "__main__":
    det_graph = load_graph()

    PATH_TO_TEST_IMAGES_DIR = '/Users/kanhua/Dropbox/Programming/udacity-carnd/CarND-Capstone/data/images/'
    image_list = ['GBTN7Ikh.PNG', 'gGG1Utds.PNG', 'qmedhAQN.PNG']
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, image_list[i]) for i in range(len(image_list))]

    detect_object(det_graph, TEST_IMAGE_PATHS)
