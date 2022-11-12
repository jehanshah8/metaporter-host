# metaporter-host

Usage: sudo docker run -it --name metaporter-host -v /<path to metaporter_dev>/metaporter_dev:/home/metaporter_dev metaporter-host:1.0

Please follow this tutorial when building and running the package:
https://docs.ros.org/en/foxy/Tutorials/Beginner-Client-Libraries/Creating-A-Workspace/Creating-A-Workspace.html

Steps to build and run the package:
(make sure to build a Workspace on /home/metaporter_dev)
Under /home/metaporter_dev directory, do:

source /home/metaporter_dev/.bashrc

source /opt/ros/foxy/setup.bash

colcon build

. install/local_setup.bash

ros2 run metaporter_host pose_subscriber
