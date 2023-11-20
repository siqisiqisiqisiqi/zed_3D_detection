#!/home/grail/.virtualenvs/3d_object_detection/bin/python
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(PARENT_DIR)

from src.params import *
from models.amodal_3D_model import Amodal3DModel
import open3d as o3d
from numpy import ndarray
import rospy
import numpy as np
import torch


class ObjectDection:
    def __init__(self):
        rospy.init_node("ObjectDection", anonymous=True)

        self.rate = rospy.Rate(10)
        self.param_fp = rospy.get_param("~param_fp")

    def downsample(self, pc_in_numpy: ndarray, num_object_points: int) -> ndarray:
        """downsample the pointcloud

        Parameters
        ----------
        pc_in_numpy : ndarray
            point cloud in adarray
            size [N, 6]
        num_object_points : int
            num of object points desired

        Returns
        -------
        ndarray
            downsampled pointcloud
        """
        pc_num = len(pc_in_numpy)
        idx = np.random.randint(pc_num, size=num_object_points)
        downsample_pc = pc_in_numpy[idx, :]
        return downsample_pc

    def point_cloud_input(self, pt_path):
        # read the pointcloud and convert the unit to centimeter
        pcd = o3d.io.read_point_cloud(pt_path)
        pc_in_numpy = np.asarray(pcd.points)
        pc_in_numpy = pc_in_numpy * 100
        # subsample the pointcloud
        pc_in_numpy = self.downsample(pc_in_numpy, NUM_OBJECT_POINT)
        pc_in_tensor = torch.tensor(pc_in_numpy)

        return torch.reshape(pc_in_tensor, (1, NUM_OBJECT_POINT, 3))

    def run(self):
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        model = Amodal3DModel()
        model.to(device)

        result_path = f"{self.param_fp}/1015_epoch40.pth"
        result = torch.load(result_path)
        model_state_dict = result['model_state_dict']
        model.load_state_dict(model_state_dict)
        model.eval()

        data_path = f"{self.param_fp}/Pointcloud2_item0.ply"
        features = self.point_cloud_input(data_path)
        features = features.to(device, dtype=torch.float)
        with torch.no_grad():
            corners = model(features)
        rospy.loginfo(f"corners value is {corners}.")

        while not rospy.is_shutdown():
            rospy.loginfo("Everything is good.")
            self.rate.sleep()


if __name__ == "__main__":
    try:
        node = ObjectDection()
        node.run()
    except rospy.ROSInterruptException:
        pass

    rospy.loginfo("Exiting camera calibration node!")
