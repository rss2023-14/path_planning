#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, PoseArray, Point
from nav_msgs.msg import Odometry, OccupancyGrid
import rospkg
import random
import time, os
# import skimage
# import skimage.morphology
import cv2
from utils import LineTrajectory
from tf.transformations import euler_from_quaternion


class PathPlan(object):
    """Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        rospy.loginfo("Initializing path planning...")
        self.graph = None
        self.occupancy = None
        self.goal = None # add so we can access goal in graph making for rrt
        self.NUM_VERTICES = rospy.get_param("num_vertices", 1000)
        self.NUM_EDGES_PER_NODE = rospy.get_param("num_edges_per_node", 20)

        self.translation = None
        self.rot_matrix = None
        self.inv_rot_matrix = None
        self.resolution = None

        self.odom_topic = rospy.get_param("odom_topic", "/pf/pose/odom")
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb)
        self.current_pose = None

        self.goal_sub = rospy.Subscriber(
            "/move_base_simple/goal", PoseStamped, self.goal_cb, queue_size=10
        )

        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_cb)
        self.trajectory = LineTrajectory("/planned_trajectory")
        self.traj_pub = rospy.Publisher("/trajectory/current", PoseArray, queue_size=10)

    def map_cb(self, msg):
        """
        Gets OccupancyGrid map and constructs graph.
        """
        # skimage.io.imsave("stata_basement_occupancy.png", msg.data)

        o = msg.info.origin.orientation
        theta = euler_from_quaternion([o.x, o.y, o.z, o.w])[2]
        sin_val = np.sin(theta)
        cos_val = np.cos(theta)

        # Get map info
        self.resolution = msg.info.resolution
        p = msg.info.origin.position
        self.translation = np.array([p.x, p.y, p.z])
        self.rot_matrix = np.matrix(
            [[cos_val, -sin_val, 0.0], [sin_val, cos_val, 0.0], [0.0, 0.0, 1.0]]
        )
        self.inv_rot_matrix = np.linalg.inv(self.rot_matrix)

        # Make graph
        (self.graph, self.occupancy) = self.make_occupancy_graph(
            np.array(msg.data), msg.info.width, msg.info.height
        )
        rospy.loginfo("Initialized, graph made!")

    def odom_cb(self, msg):
        """
        Gets Odometry msg and stores current pose.
        """
        p = msg.pose.pose.position  # x, y, z
        o = msg.pose.pose.orientation  # x, y, z, w
        t = msg.twist

        self.current_pose = (p.x, p.y)  # Only do (x,y)

    def goal_cb(self, msg):
        """
        Get goal point as PoseStamped, and initiate path planning to follow.
        """
        if self.current_pose is None:
            rospy.logwarn("Odom not initialized yet!")
            return
        elif self.graph is None:
            rospy.logwarn("Graph not initialized yet!")
            return

        # Start
        sp = self.current_pose
        so = (0, 0, 0, 0)

        # Goal
        gp = msg.pose.position  # x, y, z
        go = msg.pose.orientation  # x, y, z, w

        self.goal = (gp.x, gp.y)

        path = self.plan_path(sp, (gp.x, gp.y), self.graph)

        # Find RRT path
        path = self.make_rrt_path(
            self.occupancy, msg.info.width, msg.info.height
        )
        rospy.loginfo("rrt path found!")


    def make_rrt_path(self, data, width, height, distance_to_goal = 25, rewiring_radius = 50):
        """
        Find a path to the goal with rrt

        Args:
            data (2d array): occupancy grid data
            width (int): img width
            height (int): img height
            distance_to_goal (int): maximum distance from sampled point to actual goal point where the algo will consider the sampled point the goal
            rewiring_radius (int): maximum distance a new point will consider an old point to be rewired to

        Returns:
            list of tuples which is the path from the start to the goal
        """
        # check preconditions of having a current position and goal
        if self.current_pose is None: # start
            rospy.logwarn("Odom not initialized yet!")
            return []
        
        if self.goal is None: # end
            rospy.logwarn("Goal not initialized yet!")
            return []

        def close_to_goal(point, distance_to_goal):
            distance = ((point[0] - self.goal[0])**2 + (point[1] - self.goal[1])**2) ** 0.5
            if distance <= distance_to_goal:
                return True
            return False

        # rrt algorithm
        rrt_start = (self.current_pose[0], self.current_pose[1]) # make deep copy of start node
        rrt_graph = {rrt_start: {"parent": rrt_start, rrt_start: 0}} #{(x1, y1): {parent: (i, j), (x2, y2): distance}}
        path = None

        for i in range(10000):

            (x, y) = (random.randrange(width), random.randrange(height)) # sample point
            if data[y][x]: # if sample pointed is not in obstacle
                
                point_to_add = (x, y)
                # find which point in graph is closest
                min_dist_to_point = np.inf
                closest_point = None # will store point in graph closest to sampled point
                # close_points = [] # candidates for rewiring
                for existing_point in rrt_graph.keys(): 
                    distance = ((x - existing_point[0])**2 + (y - existing_point[1])**2) ** 0.5
                    # if distance < rewiring_radius:
                    #     close_points.append(distance)
                    if distance < min_dist_to_point:
                        min_dist_to_point = distance
                        closest_point = existing_point

                assert(closest_point is not None)
                i = closest_point[0]
                j = closest_point[1]

                # check if line between points goes through obstacle
                if False: # world frame
                    (x_img, y_img) = self.world_to_pixel(x, y)
                    (i_img, j_img) = self.world_to_pixel(i, j)
                    line_pixels = self.create_line(int(x_img), int(y_img), int(i_img), int(j_img))
                else:
                    line_pixels = self.create_line(x, y, i, j)
                
                is_collision = False
                for x_pos, y_pos in line_pixels:
                    if not data[y_pos][x_pos]:
                        is_collision = True
                        break

                if is_collision:
                    continue # sample new point

                if True: # distance in graph is just between the two points
                    rrt_graph[closest_point][point_to_add] = min_dist_to_point
                else: # distance is total distance for potential rewiring in the future
                    rrt_graph[closest_point][point_to_add] = min_dist_to_point + rrt_graph[rrt_graph[closest_point]["parent"]][closest_point]
                rrt_graph[point_to_add] = {"parent": closest_point}

                if close_to_goal(point_to_add, distance_to_goal):
                    path = [point_to_add]
                    backtrack_point = point_to_add
                    while backtrack_point != rrt_start:
                        backtrack_point = rrt_graph[backtrack_point]["parent"]
                        path.insert(0, backtrack_point)
                        #backtrack to find path to return
                    break
        
        if path is None:
            rospy.logwarn("Path not found!")
            return []
        
        world_path = []
        for x, y in path:
            world_path.append(self.pixel_to_world(x, y))

        return world_path


    def plan_path(self, start_point, end_point, map):
        """
        Take graph and use A* planning to find best path according to heuristic function.
            - Heuristic takes (start, end) nodes
            - Assumes map is a nested dictionary of neighbors {(x1, y1): {(x2, y2): distance}}
        """
        self.trajectory.clear()
        rospy.loginfo("Planning path...")

        # Define a function to calculate the Manhattan distance between two points
        def heuristic(start, end):
            return abs(start[0] - end[0]) + abs(start[1] - end[1])

        self.add_node(start_point[0],start_point[1])
        self.add_node(end_point[0],end_point[1])

        # Define a dictionary to keep track of the cost of reaching each node from the start node
        g_scores = {start_point: 0}

        # Define a dictionary to keep track of the parent node for each visited node
        parents = {start_point: None}

        # Create a set to keep track of the visited nodes
        visited = set()

        # Create a list to store the open nodes and their f-scores
        open_nodes = [(heuristic(start_point, end_point), start_point)]

        # Loop until we find the goal node or exhaust all possible paths
        i = 1 # Track iterations
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

                # Construct trajectory
                for node in path:
                    p = Point(x=node[0], y=node[1], z=0)
                    self.trajectory.addPoint(p)
                self.traj_pub.publish(self.trajectory.toPoseArray())
                self.trajectory.publish_viz()

                rospy.loginfo("Path found after " + str(i) + " iterations: " + str(path))
                return path

            # Add the current node to the visited set
            visited.add(current)

            # Loop through the current node's neighbors
            for neighbor in map[current]:
                if neighbor in visited:
                    continue
                d = np.linalg.norm(np.array(neighbor) - np.array(end_point))
                i += 1

                # Calculate the tentative g-score for this neighbor
                distance = map[current][neighbor]
                tentative_g_score = g_scores[current] + distance

                # If we have not yet visited this neighbor, or if we have found a shorter path to it,
                # update its g-score and add it to the list of open nodes
                if neighbor not in g_scores or tentative_g_score < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, end_point)
                    open_nodes.append((f_score, neighbor))
                    parents[neighbor] = current

        # No path found
        rospy.logwarn("No path found!")
        return None

    def pixel_to_world(self, x, y):
        """
        Takes position in image frame and returns same position in world frame.
        """
        # result = np.dot(self.inv_rot_matrix, ([x, y, 0] - self.translation))
        # return (result[0, 0] / self.resolution, result[0, 1] / self.resolution)

        result = np.array([x,y,0]) * self.resolution
        result = np.dot(self.rot_matrix, result)
        result += self.translation
        return (result[0,0], result[0,1])

    def world_to_pixel(self, x, y):
        """
        Takes position in world frame and returns same position in pixel frame.
        """
        # result = (
        #     np.dot(self.rot_matrix, [x * self.resolution, y * self.resolution, 0])
        #     + self.translation
        # )
        # return (result[0, 0], result[0, 1])
        result = np.array([x,y,0])
        result -= self.translation
        result = np.dot(self.inv_rot_matrix, result)
        result /= self.resolution
        return (result[0,0], result[0,1])

    def make_occupancy_graph(self, data, width, height):
        """
        Create an eroded occupancy grid and graph of the map.

        Args:
            data (1d array): occupancy grid data
            width (int): img width
            height (int): img height

        Returns:
            (dict of dicts, 2d array): graph and eroded occupancy grid
        """
        # Reset occupancy grid values
        for i in range(len(data)):
            if data[i] == -1:
                data[i] = 0
            elif data[i] > 5:
                data[i] = 0
            else:
                data[i] = 100

        """
        # Code using scikit-image replace by cv2, to avoid import errors

        # Construct image
        img = skimage.color.rgb2gray(np.array(data).reshape(height, width))
        width = len(img[0])
        height = len(img)

        # Erode occupancy grid
        globalthreshold = skimage.filters.threshold_otsu(img)
        occupancy_grid = img > globalthreshold
        footprint = skimage.morphology.disk(10)
        occupancy_grid = skimage.morphology.erosion(occupancy_grid, footprint)
        """

        img = np.array(data).reshape(height, width).astype(np.uint8) * 255
        height, width = img.shape
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Erode occupancy grid
        kernel_size = 10
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        eroded = cv2.erode(thresh, kernel)
        occupancy_grid = eroded.astype(np.bool)

        # Get random sample of valid pixel locations
        vertices = {(834, 527)}
        while len(vertices) < self.NUM_VERTICES:
            (x, y) = (random.randrange(width), random.randrange(height))
            if occupancy_grid[y][x]:
                vertices.add((x, y))

        # Create graph from vertices
        adjacency = {}
        for x, y in vertices:
            adjacency[(x, y)] = self.find_nearest_nodes(x, y, vertices, occupancy_grid, self.NUM_EDGES_PER_NODE)

        # Transform to world frame
        world_frame_adj = {}
        for x, y in adjacency:
            adjacent = {}
            for i, j in adjacency[(x, y)]:
                adjacent[self.pixel_to_world(i, j)] = (
                    adjacency[(x, y)][(i, j)] * self.resolution
                )
            world_frame_adj[self.pixel_to_world(x, y)] = adjacent

        return (world_frame_adj, occupancy_grid)

    def add_node(self, x, y):
        """
        Add node (x,y) to graph, connect edges to closest neighbors
        """
        nearest = self.find_nearest_nodes(x, y, self.graph, self.occupancy, self.NUM_EDGES_PER_NODE, True)
        self.graph[(x, y)] = nearest

        for node in nearest:
            self.graph[node][(x, y)] = nearest[node]

    def find_nearest_nodes(self, x, y, vertices, occupancy_grid, num_taken, world_frame=False):
        """
        Find nearest valid nodes in graph to (x,y).

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
                (x_img, y_img) = self.world_to_pixel(x, y)
                (i_img, j_img) = self.world_to_pixel(i, j)
                line_pixels = self.create_line(int(x_img), int(y_img), int(i_img), int(j_img))
            else:
                line_pixels = self.create_line(x, y, i, j)
            for x_pos, y_pos in line_pixels:
                if not occupancy_grid[y_pos][x_pos]:
                    remove_these.add((i, j))
                    break

        for pos in remove_these:
            del nearest[pos]

        return nearest

    def create_line(self, x, y, i, j):
        """
        Find points between two points (x,y) and (i,j).

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
