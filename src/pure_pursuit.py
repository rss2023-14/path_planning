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
from driver_pursuit import PursuitController

class PurePursuit(object):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """
    def __init__(self):
        self.odom_topic       = rospy.get_param("~odom_topic", "pf/pose/odom")
        self.lookahead        = 1.0 # FILL IN 
        # self.speed            = # FILL IN 
        # self.wheelbase_length = # FILL IN 
        self.pose = None
        self.trajectory  = utils.LineTrajectory("/followed_trajectory")
        self.traj_sub = rospy.Subscriber("/trajectory/current", PoseArray, self.trajectory_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size = 1)
        self.nearest_pt_sub = rospy.Publisher("/lookaheadpoint", PointStamped, queue_size=1)
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
        #current_traj_point = np.array([self.trajectory.points[0], self.trajectory.points[1]])

        if len(self.trajectory.points) == 0:
            rospy.loginfo("no points")
            return

        projection, min_dist_index = self.min_distance(self.trajectory.points, current_robot_position)
        lookahead = self.find_lookahead(min_dist_index, current_robot_position)
        #publish projection onto rviz 
        if lookahead is not None:
            point = Point(x = lookahead[0], y = lookahead[1], z = 0)
            self.nearest_pt_sub.publish(PointStamped(point = point, header = Header(frame_id = "map")))
        else:
            #print('nothing found')
            pass

    def min_distance(self, points, position):
        start_points = np.array(points)
        #to get distances between every adjacent pair of points
        end_points = np.roll(start_points, -1, axis=0)
        start_points = start_points[:-1]
        end_points = end_points[:-1]

        l2 = np.linalg.norm(start_points - end_points, axis=1)**2 #distances between every two pair of points
        # l2 = np.linalg.norm(start_points - end_points, axis=0)**2 #distances between every two pair of points

        #put a condition to make l2 = epsilon wherever it is equal to 0
        l2 = np.where(l2 == 0, 0.001, l2)

        rospy.logerr(np.shape(position))
        rospy.logerr(np.shape(start_points))
        rospy.logerr(np.shape(end_points))
        t = np.clip(np.sum((position - start_points) * (end_points - start_points), axis=1)/l2, 0, 1)
        projection = start_points + (t * (end_points - start_points).T).T
        distance = np.linalg.norm(position - projection, axis=1)
        minimum_dist_index = np.argmin(distance)
        min_distance = distance[minimum_dist_index]
        min_point = projection[minimum_dist_index]
        return min_point, minimum_dist_index
    
    def find_lookahead(self, minimum_dist_index, position):
        current_index = minimum_dist_index
        Q = position
        r = self.lookahead

        while True:
            P1 = np.array(self.trajectory.points[current_index])
            P2 = np.array(self.trajectory.points[current_index + 1])
            V = P2 - P1
            a = V.dot(V)
            b = 2 * V.dot(P1 - Q)
            c = P1.dot(P1) + Q.dot(Q) - 2 * P1.dot(Q) - r**2
            disc = b**2 - 4*a*c 
            if disc < 0: #no real solutions
                current_index += 1
                if current_index >= len(self.trajectory.points) - 1:
                    return None
                continue
            
            sqrt_disc = disc ** 0.5
            t1 = (-b + sqrt_disc) / (2*a)

            if not (0 <= t1 <= 1):
                #print('missed line segment')
                current_index += 1
                if current_index >= len(self.trajectory.points) - 1:
                    return None
                continue
            else:
                return P1 + t1 * V

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
    df = PursuitController()
    rospy.spin()
