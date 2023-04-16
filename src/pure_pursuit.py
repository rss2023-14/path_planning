#!/usr/bin/env python

import rospy
import numpy as np
import time
import utils
import tf

from geometry_msgs.msg import PoseArray, PoseStamped, PointStamped, Point
from visualization_msgs.msg import Marker
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Header

class PurePursuit(object):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """
    def __init__(self):
        self.odom_topic       = rospy.get_param("~odom_topic", "pf/pose/odom")
        # self.lookahead        = # FILL IN 
        # self.speed            = # FILL IN 
        # self.wheelbase_length = # FILL IN 
        self.pose = None
        self.trajectory  = utils.LineTrajectory("/followed_trajectory")
        self.traj_sub = rospy.Subscriber("/trajectory/current", PoseArray, self.trajectory_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size = 1)
        self.nearest_pt_sub = rospy.Publisher("/nearest_point", PointStamped, queue_size=1)
        self.drive_pub = rospy.Publisher("/drive", AckermannDriveStamped, queue_size=1)


    def trajectory_callback(self, msg):
        ''' Clears the currently followed trajectory, and loads the new one from the message
        '''
        print ("Receiving new trajectory:", len(msg.poses), "points")
        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)
    
    def odom_callback(self, odom):
        self.pose = odom.pose.pose
        current_robot_position = np.array([self.pose.position.x, self.pose.position.y])
        current_traj_point = np.array([self.trajectory.points[0], self.trajectory.points[1]])
        projection, distance = self.minimum_distance(current_traj_point[0], current_traj_point[1], current_robot_position)
        #publish projection onto rviz 
        point = Point(x = projection[0], y = projection[1], z = 0)
        self.nearest_pt_sub.publish(PointStamped(point = point, header = Header(frame_id = "map")))

    def minimum_distance(self, v, w, p):
        #v, w are your start points and end points 
        #p is your current position 

        l2 = np.linalg.norm(w-v)**2 #|v-w|^2 
        if l2 == 0.0: return np.linalg.norm(w-v)
    
        t = max(0, min(1, np.dot(p-v, w-v)/l2))
        projection = v + t * (w-v)
        #print(p, projection)
        distance = np.linalg.norm(p - projection)

        return projection, distance



if __name__=="__main__":
    rospy.init_node("pure_pursuit")
    pf = PurePursuit()
    rospy.spin()
