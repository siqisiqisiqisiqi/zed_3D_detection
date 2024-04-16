#!/home/grail/.virtualenvs/3d_object_detection/bin/python
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(PARENT_DIR)

import cv2
import rospy
import ros_numpy
import numpy as np
from sensor_msgs.msg import Image, PointCloud2
import time
import pickle

from src.params import *


class CameraCalib:
    def __init__(self):
        rospy.init_node("test", anonymous=True)

        # Init subscribers
        rospy.Subscriber("zed2i/zed_node/point_cloud/cloud_registered",
                         PointCloud2, self.get_pointcloud)
        self.rate = rospy.Rate(10)
        self.img_shape = (720, 1280)
        self.save_data = []

    def get_pointcloud(self, data):
        pc = ros_numpy.numpify(data)
        x = pc['x']
        y = pc['y']
        z = pc['z']
        self.points = np.stack((x, y, z), axis=2)
        # rospy.loginfo(
        #     f"the center of the camera depth is {self.points[270, 480, 2]}")
        rgb = np.zeros((self.img_shape[0], self.img_shape[1], 3))
        pc = ros_numpy.point_cloud2.split_rgb_field(pc)
        rgb[:, :, 0] = pc['r']
        rgb[:, :, 1] = pc['g']
        rgb[:, :, 2] = pc['b']
        self.color = rgb / 255.0

        self.points_z = self.points[:, :, 2]
        # print(self.points_z)
        points_z = self.points_z[~np.isnan(self.points_z)]
        z_value = np.average(points_z)
        self.save_data.append(z_value)

        # rospy.loginfo(f"the shape of z is {z.shape}")

        if len(self.save_data) == 150:
            with open(f"{PARENT_DIR}/test3", "wb") as fp:  # Pickling
                pickle.dump(self.save_data, fp)
                rospy.loginfo(
                    f"Successfully save the data!!!!!!!!!!!!!!!!!!")

    def run(self):

        while not rospy.is_shutdown():
            self.rate.sleep()


if __name__ == "__main__":
    try:
        node = CameraCalib()
        node.run()
    except rospy.ROSInterruptException:
        pass

    cv2.destroyAllWindows()
    rospy.loginfo("Finish the code!")
