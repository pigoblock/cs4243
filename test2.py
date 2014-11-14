import cv2
import cv2.cv as cv
import numpy as np
import matplotlib.pyplot as plt
from cv2 import getPerspectiveTransform, warpPerspective, imshow

img = cv2.imread("assets/project.jpeg", cv2.CV_LOAD_IMAGE_COLOR)
resImg = np.zeros((len(img), len(img[0])))
currRectImg = img[786:852, 705:750]
camera = np.matrix([[1.0, 0, 0],[0, 1.0, 0],[0, 0, 1]])
pt1 = np.float32([[-150, -20, 10]])

pt2 = [-50, -20, 10]
pt3 = [-50, 0, 10]
pt4 = [-150, 0, 10]
pt1_proj = np.zeros((3))
pt2_proj = np.zeros((3))
pt3_proj = np.zeros((3))
pt4_proj = np.zeros((3))
pt1_proj = cv2.projectPoints(pt1,(0, 0, 0), (0, 0, 0), camera, 0)