import numpy as np
import open3d as o3d
from pathlib import Path
import glob
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R


pts = glob.glob('include/data/point_cloud0.ply')


with np.load('params/E.npz') as X:
    mtx, dist, Mat, tvecs = [X[i] for i in ('mtx', 'dist', 'Mat', 'tvecs')]

# transformation from the ros camera frame to cv2 camera frame
M1 = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
# turn 180 degree around y axis
M2 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

Mat_fact = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]])
mtx = Mat_fact @ mtx

Mat2 = inv(M1)@Mat@inv(M2)
tvecs2 = inv(M1)@tvecs

for pt in pts:
    path = Path(pt)
    pcd = o3d.io.read_point_cloud(pt)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    # delete the outliers point
    center = np.array([0, 0, 0])
    radius = 1.2
    distances = np.linalg.norm(points - center, axis=1)
    points = points[distances <= radius]
    colors = colors[distances <= radius]

    # convert the frame from camera frame to IRF
    # points = (inv(Mat) @ (M1 @ points.T - tvecs)).T
    # points = (M2 @ (points.T)).T

    points = (inv(Mat2) @ (points.T - tvecs2)).T

    # visualize the coordinates
    # pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    #     size=0.4, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([pcd, mesh_frame])

# np.savez('params/E2.npz', mtx=mtx, dist = dist, Mat = Mat2, tvecs = tvecs2)
np.savez('params/E1.npz', mtx=mtx, dist = dist, Mat = Mat, tvecs = tvecs)