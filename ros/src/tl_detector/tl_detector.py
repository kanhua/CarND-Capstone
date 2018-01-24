#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math
import string
import random

STATE_COUNT_THRESHOLD = 3
print("opencv version:", cv2.__version__)


def record_image(cv_image, ref_state, save_path):
    image_saving_frequency = 0.1
    random_val = random.random()
    if random_val < image_saving_frequency:
        random_str = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(8)])

        cv2.imwrite(save_path + random_str + "_" + str(ref_state) + '.PNG', cv_image)


class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.already_print = False

        # The number of index differences that the traffic light classifier should turn on
        self.light_classifier_turn_on=300

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        # TODO delete this debugging line
        import os
        rospy.loginfo(os.getcwd())

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)


        rospy.spin()

    def check_tensorflow(self):
        """
        Test the installation of Tensorflow and its version

        :return:
        """

        import tensorflow as tff
        hello = tff.constant('Hello, TensorFlow!')
        sess = tff.Session()
        rospy.loginfo(sess.run(hello))
        rospy.loginfo(tff.__version__)

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights
        if not self.already_print:
            self.show_traffic_light_points()

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
            rospy.loginfo("published waypoints: %d", light_wp)
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
            rospy.loginfo("published waypoints: %d", self.last_wp)
        self.state_count += 1

    def get_closest_waypoint(self, pose, waypoints):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        # TODO implement

        closest_wp = self.get_closest_waypoint_from_coords(pose.pose.position.x, pose.pose.position.y, waypoints)

        return closest_wp

    def get_closest_waypoint_from_coords(self, pos_x, pos_y, waypoints):

        closest_dist = float('inf')
        closest_wp = 0

        for i in range(len(waypoints)):
            wpx = waypoints[i].pose.pose.position.x
            wpy = waypoints[i].pose.pose.position.y
            dist = math.sqrt((wpx - pos_x) ** 2 + (wpy - pos_y) ** 2)
            if dist < closest_dist:
                closest_dist = dist
                closest_wp = i

        return closest_wp

    def get_closest_stopline(self, car_pose, points):

        min_dist = float('inf')
        min_dist_index = -1
        for i in range(len(points)):
            distance = math.sqrt(
                (car_pose.pose.position.x - points[i][0]) ** 2 + (car_pose.pose.position.y - points[i][1]) ** 2)
            if distance < min_dist:
                min_dist = distance
                min_dist_index = i

        return min_dist_index

    def get_next_stopline(self, car_pose, min_dist_index, points):

        px = points[min_dist_index][0]
        py = points[min_dist_index][1]
        car_x = car_pose.pose.position.x
        car_y = car_pose.pose.position.y

        heading = math.atan2(py - car_y, px - car_x)

        x = car_pose.pose.orientation.x
        y = car_pose.pose.orientation.y
        z = car_pose.pose.orientation.z
        w = car_pose.pose.orientation.w
        euler_angles_xyz = tf.transformations.euler_from_quaternion([x, y, z, w])
        theta = euler_angles_xyz[-1]
        angle = math.fabs(theta - heading)

        if angle > math.pi / 4.0:
            min_dist_index += 1

        return min_dist_index % len(points)

    def _distance(self, p1, p2):
        x = p1.x - p2.x
        y = p1.y - p2.y
        z = p1.z - p2.z
        return math.sqrt(x * x + y * y + z * z)

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if (not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")

        save_path = "../../../data/images/"

        ref_state = light.state

        # Get classification
        # TODO change here use signal or classification
        if random.random() < 0.05:
            detected_state = self.light_classifier.get_classification(cv_image)
            self.last_state = detected_state
            if detected_state != ref_state:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                record_image(cv_image, ref_state, save_path)
                rospy.loginfo("det: %d, ref: %d", detected_state, ref_state)
        # return self.light_classifier.get_classification(cv_image)
        return self.last_state

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None

        if self.waypoints is not None:
            # List of positions that correspond to the line to stop in front of for a given intersection
            stop_line_positions = self.config['stop_line_positions']
            if (self.pose):
                car_position = self.get_closest_waypoint(self.pose, self.waypoints.waypoints)

            # TODO find the closest visible traffic light (if one exists)

            closest_traffic_light_index = self.get_closest_stopline(self.pose, stop_line_positions)

            # Determine the next coming traffic light index in the stopline_position_array
            # The system does not give the exact position of traffic light,
            # just use stop line positions as traffic light posistion
            next_traffic_light_index = self.get_next_stopline(self.pose, closest_traffic_light_index,
                                                              stop_line_positions)

            # Determine waypoint index that is closest to the next stopline/traffic light
            light_wp = self.get_closest_waypoint_from_coords(stop_line_positions[next_traffic_light_index][0],
                                                             stop_line_positions[next_traffic_light_index][1],
                                                             self.waypoints.waypoints)

            light = self.lights[next_traffic_light_index]
            # rospy.loginfo("next traffic light index: %d", light_wp)

            index_distance = 5000
            if (car_position is not None) and (light_wp is not None):
                index_distance = light_wp - car_position
                rospy.loginfo("number of indexes to the stop line: %d", index_distance)
            else:
                rospy.loginfo("no car and light position")

        if light and index_distance < self.light_classifier_turn_on:
            state = self.get_light_state(light)
            return light_wp, state
        # self.waypoints = None
        return -1, TrafficLight.UNKNOWN

    def show_traffic_light_points(self):

        for idx in range(len(self.lights)):
            rospy.loginfo("light point x,y: %f %f", self.lights[idx].pose.pose.position.x,
                          self.lights[idx].pose.pose.position.y)

        self.already_print = True


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
