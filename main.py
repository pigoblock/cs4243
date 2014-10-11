import os
import cv2
import cv2.cv as cv
import numpy as np
import matplotlib.pyplot as plt

grayscalePicture = cv2.imread("assets/project.jpeg", cv2.CV_LOAD_IMAGE_GRAYSCALE)
colorPicture = cv2.imread("assets/project.jpeg", cv2.CV_LOAD_IMAGE_COLOR)

picHeight= grayscalePicture.shape[0]
picWidth = grayscalePicture.shape[1]

    
