#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, PoseArray
from nav_msgs.msg import Odometry, OccupancyGrid
import rospkg
import time, os
from utils import LineTrajectory

class PathPlan(object):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """
    def __init__(self):
        self.odom_topic = rospy.get_param("~odom_topic")
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_cb)
        self.trajectory = LineTrajectory("/planned_trajectory")
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_cb, queue_size=10)
        self.traj_pub = rospy.Publisher("/trajectory/current", PoseArray, queue_size=10)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb)


    def map_cb(self, msg): # Occupancy Grid
        pass

        msg.data # int array, row major order, starting with 0,0, probabilties in range of 0, 100
        msg.info.width # int
        msg.info.height #int
        msg.info.origin.position # x, y, z
        msg.info.origin.orientation # x, y, z, w
        

    def odom_cb(self, msg): # Odometry
        pass ## REMOVE AND FILL IN ##

        msg.pose.pose.position # x, y, z
        msg.pose.pose.orientation # x, y, z, w
        msg.twist
        

    def goal_cb(self, msg): # PoseStamped
        pass ## REMOVE AND FILL IN ##

        msg.pose.position # x, y, z
        msg.pose.orientation # x, y, z, w

    def plan_path(self, start_point, end_point, map):
        ## CODE FOR PATH PLANNING ##

        # publish trajectory
        self.traj_pub.publish(self.trajectory.toPoseArray()) 

        # visualize trajectory Markers
        self.trajectory.publish_viz()


if __name__=="__main__":
    rospy.init_node("path_planning")
    pf = PathPlan()
    rospy.spin()
