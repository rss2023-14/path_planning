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
        ## Assume we have heuristic function called 'heuristic' which takes in (start,end)
        # Assume map is a dictionary of nodes and each of their neighbors with associated distance in a len 2 tuple

        # Define a function to calculate the Manhattan distance between two points
        def heuristic(start, end):
            return abs(start[0] - end[0]) + abs(start[1] - end[1])

        # Define a dictionary to keep track of the cost of reaching each node from the start node
        g_scores = {start_point: 0}

        # Define a dictionary to keep track of the parent node for each visited node
        parents = {start_point: None}

        # Create a set to keep track of the visited nodes
        visited = set()

        # Create a list to store the open nodes and their f-scores
        open_nodes = [(heuristic(start_point, end_point), start_point)]

        # Loop until we find the goal node or exhaust all possible paths
        while open_nodes:
            # Sort the open nodes by their f-scores (which is the sum of the g-score and the heuristic estimate)
            open_nodes.sort()

            # Get the node with the lowest f-score from the list of open nodes
            current = open_nodes.pop(0)[1]

            # If we have found the goal node, reconstruct the path and return it
            if current == end_point:
                path = []
                while current:
                    path.append(current)
                    current = parents[current]
                path.reverse()
                return path

            # Add the current node to the visited set
            visited.add(current)

            # Loop through the current node's neighbors
            for neighbor, distance in map[current]:
                # If we have already visited this neighbor, skip it
                if neighbor in visited:
                    continue

                # Calculate the tentative g-score for this neighbor
                tentative_g_score = g_scores[current] + distance

                # If we have not yet visited this neighbor, or if we have found a shorter path to it,
                # update its g-score and add it to the list of open nodes
                if neighbor not in g_scores or tentative_g_score < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, end_point)
                    open_nodes.append((f_score, neighbor))
                    parents[neighbor] = current

        # If we have exhausted all possible paths and have not found the goal node, return None
        return None



        # publish trajectory
        self.traj_pub.publish(self.trajectory.toPoseArray()) 

        # visualize trajectory Markers
        self.trajectory.publish_viz()


if __name__=="__main__":
    rospy.init_node("path_planning")
    pf = PathPlan()
    rospy.spin()
