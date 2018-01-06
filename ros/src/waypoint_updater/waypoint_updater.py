#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
import tf

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200  # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):

        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        rospy.Subscriber('/traffic_waypoint',Int32, self.traffic_cb)


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)


        # TODO: Add other member variables you need below
        # self.current_pos = None
        # self.base_waypoints = None
        self.traffic_waypoint=-1

        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.loop()
            rate.sleep()

        # rospy.spin()

    def pose_cb(self, msg):
        self.current_pose = msg
        #rospy.loginfo("reading in position seq: %d", self.current_pose.header.seq)

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
        #rospy.loginfo("reading in base_waypoints seq: %d", self.base_waypoints.header.seq)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.traffic_waypoint=msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint_index, velocity):
        waypoints[waypoint_index].twist.twist.linear.x = velocity

    def wp_distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
        for i in range(wp1, wp2 + 1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def distance(self, p1, p2):
        x = p1.x - p2.x
        y = p1.y - p2.y
        z = p1.z - p2.z
        return math.sqrt(x * x + y * y + z * z)

    def loop(self):
        if hasattr(self, 'base_waypoints') and hasattr(self, 'current_pose'):
            lane = Lane()
            lane.header.stamp = rospy.Time().now()
            lane.header.frame_id = '/world'

            pose = self.current_pose
            wpts = self.base_waypoints.waypoints

            next_wp = self.get_next_waypoint(pose, wpts)

            if self.traffic_waypoint==-1:
                lane.waypoints = self.get_final_waypoints(wpts, next_wp, next_wp + LOOKAHEAD_WPS)
                rospy.loginfo('road index:%d, %d',
                          next_wp, next_wp + LOOKAHEAD_WPS)
            else:
                lane.waypoints = self.get_final_waypoints(wpts, next_wp, self.traffic_waypoint)

                lane_length=len(lane.waypoints)
                for l in range(max((0,lane_length-30)), lane_length):
                    self.set_waypoint_velocity(lane.waypoints, l, 0)

                #rospy.loginfo('length to end point: %f', self.wp_distance(lane.waypoints,0,5))
                rospy.loginfo('road index:%d, %d',
                              next_wp, self.traffic_waypoint)
            #rospy.loginfo('first speed: %f', self.get_waypoint_velocity(lane.waypoints[0]))

            self.final_waypoints_pub.publish(lane)

    def get_closest_waypoint(self, pose, waypoints):
        closest_dist = float('inf')
        closest_wp = 0
        for i in range(len(waypoints)):
            dist = self.distance(pose.pose.position, waypoints[i].pose.pose.position)
            if dist < closest_dist:
                closest_dist = dist
                closest_wp = i

        return closest_wp

    def get_next_waypoint(self, pose, waypoints):
        closest_wp = self.get_closest_waypoint(pose, waypoints)
        wp_x = waypoints[closest_wp].pose.pose.position.x
        wp_y = waypoints[closest_wp].pose.pose.position.y
        heading = math.atan2((wp_y - pose.pose.position.y), (wp_x - pose.pose.position.x))
        x = pose.pose.orientation.x
        y = pose.pose.orientation.y
        z = pose.pose.orientation.z
        w = pose.pose.orientation.w
        euler_angles_xyz = tf.transformations.euler_from_quaternion([x, y, z, w])
        theta = euler_angles_xyz[-1]
        angle = math.fabs(theta - heading)
        if angle > math.pi / 4.0:
            closest_wp += 1

        return closest_wp

    def get_final_waypoints(self, waypoints, start_wp, end_wp):
        final_waypoints = []
        for i in range(start_wp, end_wp):
            index = i % len(waypoints)
            wp = Waypoint()
            wp.pose.pose.position.x = waypoints[index].pose.pose.position.x
            wp.pose.pose.position.y = waypoints[index].pose.pose.position.y
            wp.pose.pose.position.z = waypoints[index].pose.pose.position.z
            wp.pose.pose.orientation = waypoints[index].pose.pose.orientation
            #wp.twist.twist.linear.x = waypoints[index].twist.twist.linear.x
            wp.twist.twist.linear.x = self.get_sine_speed(index)
            final_waypoints.append(wp)

        return final_waypoints

    def get_sine_speed(self,index):

        T=3*79*2
        v_0=12.0

        v=v_0+v_0/2.0*math.sin(2.0*3.1416*float(index)/float(T))

        return v





if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
