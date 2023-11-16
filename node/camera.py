#!/home/grail/.virtualenvs/3d_object_detection/bin/python

import ros_numpy
import torch
import rospy
import cv2
from numpy.linalg import inv
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, PointCloud2
from ultralytics import YOLO
import open3d as o3d
import numpy as np
import copy


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

        param_fp = rospy.get_param("~param_fp")
        with np.load(param_fp + '/E.npz') as X:
            self.mtx, self.dist, self.Mat, self.tvecs = [
                X[i] for i in ("mtx", "dist", "Matrix", "tvec")]

        # Init the yolo model
        self.model = YOLO(param_fp + '/best.pt')

        # Init the variables
        self.points = None
        self.cv_image = None
        self.color = None
        self.shape = (540, 960)
        self.rate = rospy.Rate(10)

    def get_pointcloud(self, data):
        pc = ros_numpy.numpify(data)
        x = pc['x']
        y = pc['y']
        z = pc['z']
        # rgb = pc['rgb']
        self.points = np.stack((x, y, z), axis=2)

        # color = np.ravel(rgb).view('uint8').reshape(
        #     self.shape[0], self.shape[1], 4)
        # test = np.ravel(rgb).view('uint8')
        rgb = np.zeros((self.shape[0], self.shape[1], 3))
        pc = ros_numpy.point_cloud2.split_rgb_field(pc)
        rgb[:, :, 0] = pc['r']
        rgb[:, :, 1] = pc['g']
        rgb[:, :, 2] = pc['b']
        self.color = rgb/255.0

    def get_image(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def projection(self, point, mtx, Mat, tvecs):
        point2 = inv(mtx) @ point
        vec_z = Mat[:, [2]] * self.predefined_z
        Mat2 = np.copy(Mat)
        Mat2[:, [2]] = -1 * point2
        vec_o = -1 * (vec_z + tvecs)
        result = inv(Mat2) @ vec_o
        return result

    def run(self):
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
                for num in range(size[0]):
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

                    ################################### add the 3D object detection and localization code###########################

                    ###############################################################################################################

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(seg_p[:, :3])
                    pcd.colors = o3d.utility.Vector3dVector(rgba[:, :3])
                    o3d.visualization.draw_geometries([pcd])

            self.rate.sleep()


if __name__ == "__main__":
    try:
        node = CameraCalib()
        node.run()
    except rospy.ROSInterruptException:
        pass

    cv2.destroyAllWindows()
    rospy.loginfo("Exiting camera calibration node!")
