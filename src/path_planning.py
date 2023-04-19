#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, PoseArray
from nav_msgs.msg import Odometry, OccupancyGrid
import rospkg
import random
import time, os
import skimage
import skimage.morphology
from utils import LineTrajectory
from tf.transformations import euler_from_quaternion


class PathPlan(object):
    """Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        rospy.loginfo("initializing")
        self.graph = None
        self.occupancy = None

        self.translation = None
        self.rot_matrix = None
        self.inv_rot_matrix = None
        self.resolution = None

        self.odom_topic = rospy.get_param("~odom_topic")
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb)
        self.current_pose = None

        self.goal_sub = rospy.Subscriber(
            "/move_base_simple/goal", PoseStamped, self.goal_cb, queue_size=10
        )

        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_cb)
        self.trajectory = LineTrajectory("/planned_trajectory")
        self.traj_pub = rospy.Publisher("/trajectory/current", PoseArray, queue_size=10)
        rospy.loginfo("initialized!")

    def map_cb(self, msg):  # Occupancy Grid
        # msg.data  # int array, row major order, starting with 0,0, probabilties in range of 0, 100
        # msg.info.width  # int
        # msg.info.height  # int
        # msg.info.origin.position  # x, y, z
        # msg.info.origin.orientation  # x, y, z, w
        theta = euler_from_quaternion(msg.info.origin.orientation)[2]
        sin_val = np.sin(theta)
        cos_val = np.cos(theta)

        self.rot_matrix = np.matrix(
            [[cos_val, -sin_val, 0.0], [sin_val, cos_val, 0.0], [0.0, 0.0, 1.0]]
        )
        self.inv_rot_matrix = np.linalg.inv(self.rot_matrix)
        self.resolution = msg.info.resolution
        self.translation = msg.info.origin.position

        # skimage.io.imsave("stata_basement_occupancy.png", msg.data)
        (self.graph, self.occupancy) = self.make_occupancy_graph(
            np.array(msg.data), msg.info.width, msg.info.height
        )
        rospy.loginfo("graph made!")

    def odom_cb(self, msg):  # Odometry
        p = msg.pose.pose.position  # x, y, z
        o = msg.pose.pose.orientation  # x, y, z, w
        t = msg.twist

        self.current_pose = (p.x, p.y)  # Only do (x,y)

    def goal_cb(self, msg):  # PoseStamped
        """
        Get goal point, and initiate path planning to follow!
        """
        rospy.loginfo("got a goal!")
        if self.current_pose is None:
            rospy.loginfo("Odom not initialized yet!")
            return
        elif self.graph is None:
            rospy.loginfo("Graph not initialized yet!")
            return

        sp = self.current_pose
        so = (0, 0, 0, 0)
        gp = msg.pose.position  # x, y, z
        go = msg.pose.orientation  # x, y, z, w

        path = self.plan_path(sp, (gp.x, gp.y), self.graph)

    def create_sampled_graph(  # returns a graph in the form { (x1, y1) : [(x2, y2), (x3, y3)] }
        self, start_point, end_point, map
    ):  # sampling based method / PRM
        width = self.occupancy[
            0
        ].size  # self.occupancy is a 2d numpy array where array[y][x]
        height = self.occupancy.size / width
        # should be integer

        points = [
            start_point,
            end_point,
        ]  # put start and end point in the list in the form of (width, height) tuples
        for i in range(100):  # find random points
            x_rand_point = np.random.randint(0, width - 1)
            y_rand_point = np.random.randint(0, height - 1)
            points.append((x_rand_point, y_rand_point))

        valid_points = []
        for x, y in points:  # make sure sampled points are not inside obstacles
            if self.occupancy[y, x] == True:  # True means they are not inside obstacle
                valid_points.append((x, y))

        assert (
            start_point in valid_points
        )  # make sure start_point & end_point are in correct form
        assert end_point in valid_points

        adjacency_graph = {point: [] for point in valid_points}
        for index, first_point in enumerate(
            valid_points
        ):  # find all permutations of points
            for second_point in valid_points[index + 1 :]:
                slope = float(second_point[1] - first_point[1]) / (
                    second_point[0] - first_point[0]
                )  # find equation for line
                b = first_point[1] - slope * first_point[0]

                isCollision = False
                for x_val in range(
                    first_point[0], second_point[0]
                ):  # check to see if any discrete x values of line are in bad area
                    y_val = slope * x_val + b
                    if (
                        self.occupancy[y_val, x_val] == False
                    ):  # if one of the points on the line is in an obstacle, dont add to adjacency graph
                        isCollision = True
                        break

                if not isCollision:
                    adjacency_graph[first_point].append(second_point)
                    adjacency_graph[second_point].append(first_point)

        return adjacency_graph

    def plan_path(self, start_point, end_point, map):
        ## Assume we have heuristic function called 'heuristic' which takes in (start,end)
        # Assume map is a dictionary of nodes and each of their neighbors with associated distance in a len 2 tuple
        rospy.loginfo("planning path!")

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
            open_nodes.sort()

            current = open_nodes.pop(0)[1]

            # If we have found the goal node, reconstruct the path and return it
            if current == end_point:
                path = []
                while current:
                    path.append(current)
                    current = parents[current]
                path.reverse()
                # publish trajectory
                self.traj_pub.publish(self.trajectory.toPoseArray())

                # visualize trajectory Markers
                self.trajectory.publish_viz()
                return path

            # Add the current node to the visited set
            visited.add(current)

            # Loop through the current node's neighbors
            for neighbor, distance in map[current]:
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

        # return none if no paths are found
        # publish trajectory
        self.traj_pub.publish(self.trajectory.toPoseArray())

        # visualize trajectory Markers
        self.trajectory.publish_viz()
        return None

    def pixel_to_world(self, x, y):
        """takes position in image frame and returns same position in world frame"""
        result = np.dot(self.inv_rot_matrix, ([x, y, 0] - self.translation))

        return (result[0, 0] / self.resolution, result[0, 1] / self.resolution)

    def world_to_pixel(self, x, y):
        """takes position in world frame and returns same position in pixel frame"""
        result = (
            np.dot(self.rot_matrix, [x * self.resolution, y * self.resolution, 0])
            + self.translation
        )

        return (result[0, 0], result[0, 1])

    def make_occupancy_graph(self, data, width, height):
        """create an eroded occupancy grid and graph of the map

        Args:
            data (1d array): occupancy grid data
            width (int): img width
            height (int): img height

        Returns:
            (dict of dicts, 2d array): graph and eroded occupancy grid
        """

        for i in range(len(data)):
            if data[i] == -1:
                data[i] = 100
            elif data[i] > 5:
                data[i] = 100
            else:
                data[i] = 0

        img = skimage.color.rgb2gray(np.array(data).reshape(height, width))

        width = len(img[0])
        height = len(img)

        ## add in the start and end nodes

        # print(width, height)

        globalthreshold = skimage.filters.threshold_otsu(img)
        occupancy_grid = img > globalthreshold
        footprint = skimage.morphology.disk(10)
        occupancy_grid = skimage.morphology.erosion(occupancy_grid, footprint)

        # get random sample of valid pixel locations
        vertices = {(834, 527)}
        while len(vertices) < 1000:
            (x, y) = (random.randrange(width), random.randrange(height))
            if occupancy_grid[y][x] == False:
                vertices.add((x, y))

        # create graph from vertices
        adjacency = {}
        for x, y in vertices:
            adjacency[(x, y)] = find_nearest_nodes(x, y, vertices, occupancy_grid, 20)

        # transform to world frame
        world_frame_adj = {}
        for x, y in adjacency:
            adjacent = {}
            for i, j in adjacency[(x, y)]:
                adjacent[self.pixel_to_world(i, j)] = (
                    adjacency[(x, y)][(i, j)] * self.resolution
                )
            world_frame_adj[self.pixel_to_world(x, y)] = adjacent

        # print(adjacency)
        # skimage.io.imsave("stata_basement_thresh.png", occupancy_grid)
        return (world_frame_adj, occupancy_grid)

    def add_node(self, x, y):
        nearest = find_nearest_nodes(x, y, self.graph, self.occupancy, 20, True)

        self.graph[(x, y)] = nearest

        for node in nearest:
            self.graph[node][(x, y)] = nearest[node]


def find_nearest_nodes(x, y, vertices, occupancy_grid, num_taken, world_frame=False):
    """finds nearest valid nodes

    Args:
        x, y (float): x and y position
        vertices (iter): verts in graph
        occupancy_grid (2d array): binary occupancy grid
        num_taken (int): how many connections should we try and make

    Returns:
        dict: dict mapping near node to dist from x, y
    """
    nearest = {}
    for i, j in vertices:
        dist = ((x - i) ** 2.0 + (y - j) ** 2.0) ** 0.5
        if len(nearest) < num_taken:
            nearest[(i, j)] = dist
        else:
            cur_max = max(nearest, key=nearest.get)

            if nearest[cur_max] > dist:
                nearest[(i, j)] = dist
                del nearest[cur_max]

    del nearest[min(nearest, key=nearest.get)]

    remove_these = set()
    for i, j in nearest:
        if world_frame:
            (x_img, y_img) = PathPlan.world_to_pixel(x, y)
            (i_img, j_img) = PathPlan.world_to_pixel(i, j)
            line_pixels = create_line(int(x_img), int(y_img), int(i_img), int(j_img))
        else:
            line_pixels = create_line(x, y, i, j)
        for x_pos, y_pos in line_pixels:
            if occupancy_grid[y_pos][x_pos] == False:
                remove_these.add((i, j))
                break

    for pos in remove_these:
        del nearest[pos]

    return nearest


def create_line(x, y, i, j):
    """finds points between two points

    Args:
        x, y (float): pos 1
        i, j (float): pos 2

    Returns:
        set: points between
    """
    pixels = set()
    for x_val in range(x, i):
        y_found = (((j - y) / (i - x)) * (x_val - x)) + y
        pixels.add((x_val, int(y_found)))
    for y_val in range(y, j):
        x_found = (((i - x) / (j - y)) * (y_val - y)) + x
        pixels.add((int(x_found), y_val))

    return pixels


if __name__ == "__main__":
    rospy.init_node("path_planning")
    pf = PathPlan()
    rospy.spin()
