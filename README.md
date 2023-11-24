# zed_3D_detection
This is the repo for zed 3D detection using customized approach in ros framework.

## Table of contents

- Camera Calibration
- Run the project
- Others

## Camera Calibration

### Camera calibration in opencv frame
Put the checkboard on the IRF plane, and run the following command.

```bash
roslaunch zed_3d_detection calibration.launch
```
After running the calibration file, "E.npz" will be automatically created. It contains
the camera intrinsic and extrinsic parameters. Remember, all the parameters are calculated 
in the "opencv" defined frame.

### Camera parameter transformation to the ROS frame
ROS frame is different from the opencv frame (the camera frame is different). 
To visualize the calibration result and convert the camera parameters in the ROS frame. 
Run "processed_point_cloud.py" code and set "OPTION" to "E1". 
"E1.npz" will be automatically created which defines the camera parameters in the ROS 
frame. 

### Rotate the Z axis in IRF to the opposite direction
Traditionally, z axis direction in IRF after calibration is downward. 
Run "processed_point_cloud.py" code and set "OPTION" to "E2".
"E2.npz" will be automatically created which converts the Z axis upward.