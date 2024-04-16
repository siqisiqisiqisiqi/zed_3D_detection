import cv2
import numpy as np

image = cv2.imread("titan_5-layer.png",cv2.IMREAD_UNCHANGED)
half = cv2.resize(image,(0,0) ,fx = 0.5,fy = 0.5, interpolation= cv2.INTER_LINEAR)
print(image.shape)
cv2.imwrite("type2.png", half) 