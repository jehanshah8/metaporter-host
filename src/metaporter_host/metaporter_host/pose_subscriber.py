#!/usr/bin/env/python3
import os 
import datetime 

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped # vo_pose class

data_path =  "./src/metaport_host/recv_data/"

class PoseSubscriberNode(Node):

    def __init__(self) -> None:
        super().__init__("pose_subscriber")
        self.pose_filename = str(datetime.datetime.now()) + ".txt"
        #self.fptr = open(os.path.join(data_path,self.pose_filename), "w")
        self.fptr = open(self.pose_filename, "w+")
        self.pose_subscriber = self.create_subscription(PoseStamped, "/visual_slam/tracking/vo_pose", self.pose_callback,10)
        self.get_logger().info("hello-world")
        self.get_logger().info(str(os.getcwd()))
    
    def pose_callback(self, msg: PoseStamped):
        pose_msg = str(msg)
        self.fptr.write(pose_msg)
    
    def __del__(self):
        #self.fptr.close()
        pass
        

def main (args=None):
    rclpy.init(args=args)
    node = PoseSubscriberNode()
    rclpy.spin(node)

    

    rclpy.shutdown()


if __name__ == '__main__':
    main()
