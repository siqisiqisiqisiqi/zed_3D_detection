#!/home/grail/.virtualenvs/3d_object_detection/bin/python

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
import numpy as np


class CameraExCalib:
    def __init__(self):
        rospy.init_node("camera_excalib", anonymous=True)

        self.bridge = CvBridge()

        # Init subscribers
        rospy.Subscriber("zed2i/zed_node/rgb/image_rect_color",
                         Image, self.get_image)

        camera_info = rospy.wait_for_message(
            "zed2i/zed_node/rgb/camera_info", CameraInfo)

        rospy.sleep(1)
        self.mtx = np.array(camera_info.K).reshape(3, 3)
        self.dist = np.array([[0, 0, 0, 0, 0]]).astype("float64")

        self.rvecs = None
        self.tvecs = None

        self.param_fp = rospy.get_param("~param_fp")

        self.chess_size = 1

    def get_image(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def coordinates_draw(self, img, corners, imgpts):
        corner = tuple(corners[0].astype(int).ravel())
        cv2.line(img, corner, tuple(
            imgpts[0].astype(int).ravel()), (255, 0, 0), 5)
        cv2.line(img, corner, tuple(
            imgpts[1].astype(int).ravel()), (0, 255, 0), 5)
        cv2.line(img, corner, tuple(
            imgpts[2].astype(int).ravel()), (0, 0, 255), 5)
        return img

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
            rospy.loginfo("Complete the calibration.")
            cv2.imshow('img', img)
            cv2.waitKey(0)
        else:
            rospy.loginfo("Can't detect the checkboard!!!!!!!!!!!!!!!!!!!!!")
            rospy.loginfo("Complete the calibration.")
            cv2.imshow('img', gray)
            cv2.waitKey(0)

    def run(self):

        rospy.sleep(1)
        self.extrinsic_calibration()
        if self.rvecs is not None:
            Mat, _ = cv2.Rodrigues(self.rvecs)
            tvec = self.tvecs * self.chess_size

            np.savez(self.param_fp + '/E.npz', mtx=self.mtx,
                     dist=self.dist, Matrix=Mat, tvec=tvec)
            rospy.loginfo("Successfully save the calibration parameters.")
        rospy.sleep(2)


if __name__ == "__main__":
    try:
        node = CameraExCalib()
        node.run()
    except rospy.ROSInterruptException:
        pass

    cv2.destroyAllWindows()
    rospy.loginfo("Exiting camera Extrinsic calibration node!")
