from operator import itemgetter
import numpy as np
from numpy.linalg.linalg import inv
import cv2
import cv2.cv as cv
from cv2 import getPerspectiveTransform, warpPerspective, imshow, findHomography

def overlayImage(resultImage, homoImage):
    greyHomo = cv2.cvtColor(homoImage, cv2.cv.CV_BGR2GRAY)
    retVal, greyHomo = cv2.threshold(greyHomo, 0,255,cv2.cv.CV_THRESH_BINARY)
    greyHomo_inv = ~greyHomo
    
    newResultImage = np.zeros((picHeight, picWidth, 3), np.uint8)
    newResultImage[:] = (0, 0, 0)
    newHomoImage = np.zeros((picHeight, picWidth, 3), np.uint8)
    newHomoImage[:] = (0, 0, 0)
    
    cv2.bitwise_and(resultImage, resultImage, newResultImage,greyHomo_inv)
    cv2.bitwise_and(homoImage, homoImage, newHomoImage,greyHomo)

    
    resultImage = newResultImage + newHomoImage
    cv2.imwrite("NOOO2.jpg", resultPicture)
    cv2.imwrite("YESSSSSS2.jpg", resultImage)
    return resultImage
    
    
image = cv2.imread("project.jpeg", cv2.CV_LOAD_IMAGE_COLOR)

#Gets image pixels from xStart to xEnd -1, yStart to yEnd - 1
def cropImage(image, xStart, xEnd, yStart, yEnd):
    cropped = image[yStart:yEnd, xStart:xEnd]
    return cropped

#Gets Corners of 2D image in clockwise direction
def getCorners(image):
    return [[0,0],[len(image[0]), 0], [len(image[0]), len(image)], [0, len(image)]]

#Gets 4 projected points of a rectangle in 3D space
def getProjectedPoints(points_3D, Rvect, Tvect, camera, distVect):
    resultPoints = cv2.projectPoints(pointsMatrix, (0,0,0), (0,0,0), camera, 0)    
    resultPoints2 = [[resultPoints[0][0][0][0], resultPoints[0][0][0][1]],
                 [resultPoints[0][1][0][0], resultPoints[0][1][0][1]],
                 [resultPoints[0][2][0][0], resultPoints[0][2][0][1]],
                 [resultPoints[0][3][0][0], resultPoints[0][3][0][1]]]
    return resultPoints2

#Points should be in np.array float32 format, Returns the warped picture with black BG on ouputImgSize dimensions
def warpHomo(imgPts_2D, projPts_2D, outputImgSize):
    H = findHomography(imgPts_2D, projPts_2D , 0 );
    resultPicture = warpPerspective(currRectImg, H[0], outputImgSize)
    return resultPicture

array_2d = []
currRectImg = image[812:856, 848:960]
point_2d_1 = [0, 0]
point_2d_2 = [112, 0]
point_2d_3 = [112, 44]
point_2d_4 = [0, 44]
array_2d.append(point_2d_1)
array_2d.append(point_2d_2)
array_2d.append(point_2d_3)
array_2d.append(point_2d_4)

array_3d = []
'''
point_3d_1 = [50.0, -30, 40]
point_3d_2 = [50.0, -30, 30]
point_3d_3 = [50.0, 0, 30]
point_3d_4 = [50.0, 0, 40]
'''
picHeight= image.shape[0]
picWidth = image.shape[1]

point_3d_1 = [30.0, -60, 10]
point_3d_2 = [30.0, -60, 100]
point_3d_3 = [30.0, 0, 100]
point_3d_4 = [30.0, 0, 10]

array_3d.append(point_3d_1)
array_3d.append(point_3d_2)
array_3d.append(point_3d_3)
array_3d.append(point_3d_4)

pointsMatrix = np.float32([array_3d])

camera = np.matrix([[300.0, 0, picWidth/2],[0, 300.0, picHeight/2],[0, 0, 1]])
resultPoints = cv2.projectPoints(pointsMatrix, (0,0,0), (0,0,0), camera, 0)

resultPoints2 = [[resultPoints[0][0][0][0], resultPoints[0][0][0][1]],
                 [resultPoints[0][1][0][0], resultPoints[0][1][0][1]],
                 [resultPoints[0][2][0][0], resultPoints[0][2][0][1]],
                 [resultPoints[0][3][0][0], resultPoints[0][3][0][1]]
                 ]
'''
resultPoints2 = [[int(picHeight - (picHeight - resultPoints[0][0][0][1])), int(picWidth - (picWidth - resultPoints[0][0][0][0]))],
                 [int(picHeight - (picHeight - resultPoints[0][1][0][1])), int(picWidth - (picWidth - resultPoints[0][1][0][0]))],
                 [int(picHeight - (picHeight - resultPoints[0][2][0][1])), int(picWidth - (picWidth - resultPoints[0][2][0][0]))],
                 [int(picHeight - (picHeight - resultPoints[0][3][0][1])), int(picWidth - (picWidth - resultPoints[0][3][0][0]))]]
'''
print pointsMatrix
print resultPoints[0]

#resultPoints2 = [[0,0],
#                 [100,0],
#                 [100,100],
#                 [0,100]]

H = findHomography(np.array(array_2d, np.float32), np.array(resultPoints2, np.float32) , 0 );



resultPicture = warpPerspective(currRectImg, H[0], (picWidth,picHeight))
overlayImage(image, resultPicture)
finalPicture = resultPicture + image
