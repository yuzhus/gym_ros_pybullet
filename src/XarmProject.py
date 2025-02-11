#!/usr/bin/env python3
'''
    The script that control multiple roslaunch file for runing the digital twin online training.
'''

import roslaunch
import rospy

rospy.init_node('xarm_project', anonymous=True)
uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
roslaunch.configure_logging(uuid)
launch = roslaunch.parent.ROSLaunchParent(uuid, ["/home/xarm/xarm_catkinws/src/gym_ros_pybullet/launch/xarm_train.launch"])
launch.start()
rospy.loginfo("started")

rospy.sleep(3)
# 3 seconds later
launch.shutdown()