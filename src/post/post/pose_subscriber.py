#!/usr/bin/env/python3
import os
import datetime 

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped # vo_pose class

data_path =  "/home/metaporter_dev/src/data/"

class PoseSubscriberNode(Node):

    def __init__(self) -> None:
        super().__init__("pose_subscriber")
        self.pose_filename = datetime.date.now() + ".txt"
        self.fptr = open(os.path.join(data_path,self.pose_filename), "a")
        self.pose_subscriber = self.create_subscription(PoseStamped, "/visual_slam/tracking/vo_pose", self.pose_callback)
        self.get_logger().info("hello-world")
    
    def pose_callback(self, msg: PoseStamped):
        pose_msg = str(msg)
        self.fptr.write(pose_msg)
    
    def __del__(self):
        self.fptr.close()
        

def main (args=None):
    rclpy.init(args=args)
    node = PoseSubscriberNode()
    rclpy.spin(node)

    

    rclpy.shutdown()
