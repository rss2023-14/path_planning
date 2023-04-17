import rospy
import numpy as np

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
import tf.transformations as tf

class PursuitController():
    """
    A controller for driving to lookahead point.
    Listens for a look ahead point and publishes control commands.
    Can be used in the simulator and on the real robot.
    """

    def __init__(self):
        # rospy.loginfo("init!")

        # set in launch file; different for simulator vs racecar
        DRIVE_TOPIC = rospy.get_param("~drive_topic", "/drive")
        ODOM_TOPIC = rospy.get_param("~odom_topic", "/pf/pose/odom")
        self.lookahead_sub = rospy.Subscriber("/lookaheadpoint", PointStamped, self.look_ahead_callback)
        self.odom_sub = rospy.Subscriber(ODOM_TOPIC, Odometry, self.odom_callback)
        self.drive_pub = rospy.Publisher(DRIVE_TOPIC, AckermannDriveStamped, queue_size=1)
        
        # self.error_pub = rospy.Publisher("/parking_error",
        #                                  ParkingError, queue_size=10)

        self.parking_distance = 0 # meters; try playing with this number!
        self.relative_x = 0
        self.relative_y = 0

        self.prev_time = rospy.Time.now()

        self.prev_theta_err = 0.0
        self.prev_dist_err = 0.0

        self.running_theta_err = 0.0
        self.running_dist_err = 0.0
        self.pose = None

    def odom_callback(self, msg):
        self.pose = msg.pose.pose

    def look_ahead_callback(self, msg):
        time = rospy.Time.now()

        dt = (time - self.prev_time).to_sec()

        self.prev_time = time

        #convert frames 
        #get angle from quaternion 
        quaternion = self.pose.orientation
        #print(quaternion)
        theta = tf.euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])[2]
        rotatation_matrix = np.array([[np.cos(theta), np.sin(theta), 0],
                                      [-np.sin(theta), np.cos(theta), 0],
                                      [0, 0 , 1]])
        
        current = np.array([msg.point.x - self.pose.position.x, msg.point.y - self.pose.position.y, 0])
        transformed = np.matmul(rotatation_matrix, current)
        self.relative_x = transformed[0]
        self.relative_y = transformed[1]

        dist = np.sqrt((self.relative_x**2.0)+(self.relative_y**2.0))
        dist_err = dist

        # pos theta to left
        theta_err = np.arctan2(self.relative_y, self.relative_x)

        self.running_theta_err += theta_err
        self.running_dist_err += dist_err

        d_theta_dt = (theta_err - self.prev_theta_err) / dt
        d_dist_dt = (dist_err - self.prev_dist_err) / dt

        self.prev_dist_err = dist_err
        self.prev_theta_err = theta_err

        drive_cmd = AckermannDriveStamped()
        drive_cmd.header.stamp = rospy.Time.now()
        #drive_cmd.header.frame_id = "base_link"

        drive_cmd.drive.steering_angle = 0.4 * theta_err + \
            0.01 * d_theta_dt + 0.0 * self.running_theta_err

        drive_cmd.drive.speed = 2.0

        drive_cmd.drive.steering_angle_velocity = 0.0
        drive_cmd.drive.acceleration = 0.0
        drive_cmd.drive.jerk = 0.0

        self.drive_pub.publish(drive_cmd)
        #self.error_publisher()

    # def error_publisher(self):
    #     """
    #     Publish the error between the car and the cone. We will view this
    #     with rqt_plot to plot the success of the controller
    #     """
    #     error_msg = ParkingError()

    #     error_msg.x_error = self.relative_x
    #     error_msg.y_error = self.relative_y
    #     error_msg.distance_error = np.sqrt(
    #         (self.relative_x**2.0)+(self.relative_y**2.0))

    #     self.error_pub.publish(error_msg)
