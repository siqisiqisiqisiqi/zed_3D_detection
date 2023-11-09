#!/home/grail/.virtualenvs/3d_object_detection/bin/python

import cv2
import rospy
import torch
import numpy as np
from numpy.linalg import inv
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from ultralytics import YOLO


class CameraCalib:
    def __init__(self):
        rospy.init_node("camera", anonymous=True)

        self.bridge = CvBridge()

        # Init subscribers
        rospy.Subscriber("zed2i/zed_node/rgb/image_rect_color",
                         Image, self.get_image)

        camera_info = rospy.wait_for_message(
            "zed2i/zed_node/rgb/camera_info", CameraInfo)

        rospy.sleep(1)
        self.mtx = np.array(camera_info.K).reshape(3, 3)
        self.dist = np.array([[0, 0, 0, 0, 0]]).astype("float64")

        param_fp = rospy.get_param("~param_fp")
        np.savez(param_fp + '/B.npz', mtx=self.mtx, dist=self.dist)

        # Init the yolo model
        self.model = YOLO(param_fp + '/best.pt')
        # Init the publish rate
        self.rvecs = None
        self.tvecs = None
        self.rate = rospy.Rate(10)
        self.chess_size = 1

        # self.predefined_z = -1 * 100 / 23 / 2
        # # self.predefined_z = 0

    def get_image(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        # cv2.imshow("Image window", cv_image)
        # cv2.waitKey(3)

    def coordinates_draw(self, img, corners, imgpts):
        corner = tuple(corners[0].astype(int).ravel())
        cv2.line(img, corner, tuple(
            imgpts[0].astype(int).ravel()), (255, 0, 0), 5)
        cv2.line(img, corner, tuple(
            imgpts[1].astype(int).ravel()), (0, 255, 0), 5)
        cv2.line(img, corner, tuple(
            imgpts[2].astype(int).ravel()), (0, 0, 255), 5)
        return img

    def drawing(self, img, point_list, result_list):
        axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
        origin = np.float32([[0, 0, 0]]).reshape(-1, 3)
        imgpts, _ = cv2.projectPoints(
            axis, self.rvecs, self.tvecs, self.mtx, self.dist)
        corner, _ = cv2.projectPoints(
            origin, self.rvecs, self.tvecs, self.mtx, self.dist)
        corner = tuple(corner[0].astype(int).ravel())
        rospy.loginfo(f"the shape of the image is {img.shape}")
        img = cv2.line(img, corner, tuple(
            imgpts[0].astype(int).ravel()), (255, 0, 0), 5)
        img = cv2.line(img, corner, tuple(
            imgpts[1].astype(int).ravel()), (0, 255, 0), 5)
        img = cv2.line(img, corner, tuple(
            imgpts[2].astype(int).ravel()), (0, 0, 255), 5)
        cv2.putText(img, 'X axis', (1000, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, 'Y axis', (1150, 350),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        if len(result_list) != 0:
            for i in range(len(result_list)):
                result = result_list[i]
                point = point_list[i]
                cv2.putText(img, f'[{result[0,0]}, {result[1,0]}]', (point[0, 0] + 50, point[1,
                            0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 0, 128), 2, cv2.LINE_AA)
                cv2.circle(img, (point[0, 0], point[1, 0]),
                           5, (128, 0, 128), 5)
        cv2.imshow('img', img)
        cv2.waitKey(3)

    def projection(self, point, mtx, Mat, tvecs):
        point2 = inv(mtx) @ point
        vec_z = Mat[:, [2]] * self.predefined_z
        Mat2 = np.copy(Mat)
        Mat2[:, [2]] = -1 * point2
        vec_o = -1 * (vec_z + tvecs)
        result = inv(Mat2) @ vec_o
        return result

    def extrinsic_calibration(self):
        img = self.cv_image
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        if ret == True:
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)

            # Find the rotation and translation vectors.
            ret, self.rvecs, self.tvecs = cv2.solvePnP(
                objp, corners2, self.mtx, self.dist)

            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(
                axis, self.rvecs, self.tvecs, self.mtx, self.dist)

            img = self.coordinates_draw(img, corners2, imgpts)
            rospy.loginfo("complete the calibration")
            cv2.imshow('img', img)
            cv2.waitKey(0)

    def run(self):

        rospy.sleep(1)
        self.extrinsic_calibration()
        Mat, _ = cv2.Rodrigues(self.rvecs)
        tvec = self.tvecs * self.chess_size

        rospy.sleep(2)
        while not rospy.is_shutdown():
            result_list = []
            point_list = []
            img = np.copy(self.cv_image)
            results = self.model(img)

            # for r in results:
            #     xyxy = r.boxes.xyxy.cpu().detach().numpy()
            #     break
            # try:
            #     size = xyxy.shape
            #     num = size[0]
            #     for i in range(num):
            #         xmin = int(xyxy[i, 0])
            #         xmax = int(xyxy[i, 2])
            #         ymin = int(xyxy[i, 1])
            #         ymax = int(xyxy[i, 3])
            #         xcenter = int((xmin + xmax) / 2)
            #         ycenter = int((ymin + ymax) / 2) - 45
            #         point = np.array([[xcenter, ycenter, 1]]).T
            #         result = self.projection(point, self.mtx, Mat, tvec)
            #         result = np.round(result, 2)
            #         result_list.append(result)
            #         point_list.append(point)
            # except:
            #     rospy.loginfo("Not detect the peach!")

            # self.drawing(img, point_list, result_list)

            self.rate.sleep()


if __name__ == "__main__":
    try:
        node = CameraCalib()
        node.run()
    except rospy.ROSInterruptException:
        pass

    cv2.destroyAllWindows()
    rospy.loginfo("Exiting camera calibration node!")
