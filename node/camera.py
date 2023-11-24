#!/home/grail/.virtualenvs/3d_object_detection/bin/python
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(PARENT_DIR)

import copy
import cv2
import rospy
import torch
import ros_numpy
import numpy as np
from numpy import ndarray
from numpy.linalg import inv
import open3d as o3d
from ultralytics import YOLO
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge, CvBridgeError

from models.amodal_3D_model import Amodal3DModel
from zed_3D_detection.msg import Box3d, Corners
from src.params import *


class CameraCalib:
    def __init__(self):
        rospy.init_node("camera", anonymous=True)

        self.bridge = CvBridge()

        # Init subscribers
        rospy.Subscriber("zed2i/zed_node/rgb/image_rect_color",
                         Image, self.get_image)

        # Init subscribers
        rospy.Subscriber("zed2i/zed_node/point_cloud/cloud_registered",
                         PointCloud2, self.get_pointcloud)

        self.corner_pub2 = rospy.Publisher(
            'corners_test', Box3d, queue_size=10)

        self.param_fp = rospy.get_param("~param_fp")
        with np.load(self.param_fp + '/E1.npz') as X:
            self.mtx, self.dist, self.Mat, self.tvecs = [
                X[i] for i in ("mtx", "dist", "Mat", "tvec")]

        # Init the yolo model
        self.model = YOLO(self.param_fp + '/best.pt')

        # Init the 3D model detection model
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model3D = Amodal3DModel()
        self.model3D.to(self.device)

        # Init the variables
        self.points = None
        self.cv_image = None
        self.color = None
        # self.shape = (540, 960)
        self.shape = (720, 1280)
        self.rate = rospy.Rate(10)

    def get_pointcloud(self, data):
        pc = ros_numpy.numpify(data)
        x = pc['x']
        y = pc['y']
        z = pc['z']
        self.points = np.stack((x, y, z), axis=2)
        # rospy.loginfo(
        #     f"the center of the camera depth is {self.points[270, 480, 2]}")
        rgb = np.zeros((self.shape[0], self.shape[1], 3))
        pc = ros_numpy.point_cloud2.split_rgb_field(pc)
        rgb[:, :, 0] = pc['r']
        rgb[:, :, 1] = pc['g']
        rgb[:, :, 2] = pc['b']
        self.color = rgb / 255.0

    def get_image(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def downsample(self, pc_in_numpy: ndarray, num_object_points: int) -> ndarray:
        pc_num = len(pc_in_numpy)
        idx = np.random.randint(pc_num, size=num_object_points)
        downsample_pc = pc_in_numpy[idx, :]
        return downsample_pc

    def point_cloud_input(self, pt):
        try:
            # frame translation from camera frame to IRF frame
            pt_irf = (inv(self.Mat) @ (pt.T - self.tvecs)).T
            # read the pointcloud and convert the unit to centimeter
            pc_in_numpy = pt_irf
            pc_in_numpy = pc_in_numpy * 100
            # subsample the pointcloud
            pc_in_numpy = self.downsample(pc_in_numpy, NUM_OBJECT_POINT)
            pc_in_tensor = torch.tensor(pc_in_numpy)
            return torch.reshape(pc_in_tensor, (1, NUM_OBJECT_POINT, 3))
        except RuntimeWarning:
            return None
        except:
            return None

    def run(self):

        # Load the weight of the model
        result_path = f"{self.param_fp}/1015_epoch40.pth"
        result = torch.load(result_path)
        model_state_dict = result['model_state_dict']
        self.model3D.load_state_dict(model_state_dict)
        self.model3D.eval()
        # Need at least 4 sec to initialize the camera
        rospy.sleep(4)

        while not rospy.is_shutdown():
            img = np.copy(self.cv_image)
            results = self.model(img)

            # visualize the segment result
            # for r in results:
            #     im_array = r.plot()
            #     cv2.imshow("image", im_array)
            #     cv2.waitKey(2)

            if results[0].masks is not None:
                mask_result = results[0].masks.data.cpu().detach().numpy()
                size = mask_result.shape
                corner_data_send = Box3d()
                corner_data_send.num = size[0]

                for num in range(size[0]):

                    ################################### Crop the pointcloud according to the segmentation###########################
                    conf = results[0].boxes.conf[num].cpu().detach().numpy()
                    if conf < 0.9:
                        continue

                    mask = mask_result[num]
                    mask = cv2.resize(mask, (self.shape[1], self.shape[0]),
                                      interpolation=cv2.INTER_LINEAR)
                    logits = np.nonzero(mask < 1)

                    pt_data = copy.copy(self.points)
                    pt_data[logits[0], logits[1]] = [
                        float("nan"), float("nan"), float("nan")]

                    pt_data = pt_data.reshape(-1, 3)
                    color = self.color.reshape(-1, 3)
                    seg_p = pt_data[~np.isnan(pt_data[:, 0]), :]
                    rgba = color[~np.isnan(pt_data[:, 0]), :]

                    ################################### 3D bounding box estimation###########################
                    features = self.point_cloud_input(seg_p)
                    if features is None:
                        continue
                    features = features.to(self.device, dtype=torch.float)
                    with torch.no_grad():
                        corners = self.model3D(features)

                    corner_to_send_test = Corners()
                    corner_to_send_test.data = np.ravel(corners[0]).tolist()
                    corner_data_send.corners_data.append(corner_to_send_test)

                self.corner_pub2.publish(corner_data_send)

        ################################################ Cropped Pointcloud visualization###############################################################

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(seg_p[:, :3])
            # pcd.colors = o3d.utility.Vector3dVector(rgba[:, :3])
            # o3d.visualization.draw_geometries([pcd])
            self.rate.sleep()

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(self.points.reshape(-1, 3))
        # pcd.colors = o3d.utility.Vector3dVector(self.color.reshape(-1, 3))
        # o3d.io.write_point_cloud(
        #     f"{PARENT_DIR}/include/data/point_cloud0.ply", pcd)
        # rospy.sleep(2)


if __name__ == "__main__":
    try:
        node = CameraCalib()
        node.run()
    except rospy.ROSInterruptException:
        pass

    cv2.destroyAllWindows()
    rospy.loginfo("Finish the code!")
