#!/home/grail/.virtualenvs/3d_object_detection/bin/python
import cv2
import rospy
import numpy as np
from numpy.linalg import inv
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge, CvBridgeError


class visualization:
    def __init__(self):
        rospy.init_node("Bounding_Box_Visual", anonymous=True)

        self.bridge = CvBridge()

        rospy.sleep(3)

        # Init image subscribers
        rospy.Subscriber("zed2i/zed_node/rgb/image_rect_color",
                         Image, self.get_image)

        # Init corners subscribers
        rospy.Subscriber("/corners", Float64MultiArray, self.get_corners)

        # Get the calibration parameters
        self.param_fp = rospy.get_param("~param_fp")
        with np.load(self.param_fp + '/E2.npz') as X:
            self.mtx, self.dist, self.Mat, self.tvecs = [
                X[i] for i in ("mtx", "dist", "Mat", "tvec")]

        # Define the color used to visualized
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                       (0, 255, 255), (255, 0, 255)]

        self.rate = rospy.Rate(10)

    def get_image(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            rospy.loginfo(f"the shape of the image is {self.cv_image.shape}.")
        except CvBridgeError as e:
            print(e)

    def get_corners(self, data):
        try:
            self.corners = np.array(data.data).reshape((8, 3))
            # convert centimeter to meter
            # self.corners = corners / 100.0

            # self.corners = np.array([[5, 20, 10], [5, 13, 10], [-2, 13, 10], [-2, 20, 10], [
            #                         5, 20, 0], [5, 13, 0], [-2, 13, 0], [-2, 20, 0]])
            self.corners = self.corners / 100
            # rospy.loginfo(f" the corners value is {self.corners}")

        except:
            rospy.loginfo(f"Not detect the peach.")

    def visualization(self):

        img = self.cv_image
        corner_world = self.corners
        corner_camera = self.Mat @ (corner_world.T) + self.tvecs
        corner_image = (self.mtx @ corner_camera).T
        corner = corner_image[:, :2] / corner_image[:, 2:3]
        corner = corner.astype(int)
        rospy.loginfo(f"the corner value in pixel is {corner}")
        # corner[:, 0] = 770 - corner[:, 0]
        # corner[:, 1] = corner[:, 1] - 70

        corner1 = corner[:4, :]
        corner2 = corner[4:8, :]
        pt1 = corner1.reshape((-1, 1, 2))
        pt2 = corner2.reshape((-1, 1, 2))
        # rospy.loginfo(f"the corners are {corner}")

        index = 0
        color = self.colors[index]
        thickness = 2
        cv2.polylines(img, [pt1], True, color, thickness)
        cv2.polylines(img, [pt2], True, color, thickness)
        for i, j in zip(range(4), range(4, 8)):
            cv2.line(img, tuple(corner[i]), tuple(corner[j]), color, thickness)

        # option 2 drawing
        index1 = [1, 0, 4, 5]
        index2 = [0, 3, 7, 4]
        index3 = [2, 3, 7, 6]
        index4 = [1, 2, 6, 5]
        zero1 = np.zeros((img.shape), dtype=np.uint8)
        zero2 = np.zeros((img.shape), dtype=np.uint8)
        zero3 = np.zeros((img.shape), dtype=np.uint8)
        zero4 = np.zeros((img.shape), dtype=np.uint8)
        zero_mask1 = cv2.fillConvexPoly(zero1, corner[index1, :], color)
        zero_mask2 = cv2.fillConvexPoly(zero2, corner[index2, :], color)
        zero_mask3 = cv2.fillConvexPoly(zero3, corner[index3, :], color)
        zero_mask4 = cv2.fillConvexPoly(zero4, corner[index4, :], color)
        zeros_mask = np.array(
            (zero_mask1 + zero_mask2 + zero_mask3 + zero_mask4))

        alpha = 1
        beta = 0.55
        gamma = 0
        mask_img = cv2.addWeighted(img, alpha, zeros_mask, beta, gamma)
        cv2.imshow("Image", mask_img)
        cv2.waitKey(5)

    def run(self):
        rospy.sleep(7)
        while not rospy.is_shutdown():
            try:
                self.visualization()
            except:
                rospy.loginfo("Have not detected the peach.")
            # self.visualization()
            self.rate.sleep()


if __name__ == "__main__":
    try:
        node = visualization()
        node.run()
    except rospy.ROSInterruptException:
        pass

    cv2.destroyAllWindows()
    rospy.loginfo("Exiting camera Extrinsic calibration node!")
