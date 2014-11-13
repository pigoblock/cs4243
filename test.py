from operator import itemgetter
import numpy as np
from numpy.linalg.linalg import inv
import cv2
import cv2.cv as cv
from cv2 import getPerspectiveTransform, warpPerspective, imshow, findHomography

'''
x_screen = 705
y_screen = 785
focal_length = 10.0
center_of_projection_x = 816.0
center_of_projection_y = 612.0
distance_from_camera = 100.0
scale_factor = 1

x_world = int((x_screen - center_of_projection_x) * distance_from_camera / focal_length * scale_factor)
y_world = int((y_screen - center_of_projection_y) * distance_from_camera / focal_length * scale_factor)
z_world = distance_from_camera

print x_world,
print y_world,
print z_world

x_screen = (x_world / z_world * focal_length / scale_factor) + center_of_projection_x
y_screen = (y_world / z_world * focal_length / scale_factor) + center_of_projection_y

print x_screen,
print y_screen

array = [ [1,2,3,4,5,6],
          [6,5,4,3,2,1],
          [5,10,15,2,1,3],
          [2,2,2,2,2,2]]
array = [array[i][0:3] for i in range(0,len(array))]
print array

img = cv2.imread("assets/project.jpeg", cv2.CV_LOAD_IMAGE_COLOR)
#source = np.array([[705, 780],[835, 780],[835, 855],[705, 855]],np.float32)
source = np.array([[1175, 700],[1235, 675],[1235, 885],[1175, 875]],np.float32)

destination = np.array([[0,0],[1000,0],[1000,500],[0,500]], np.float32)
destination = destination.reshape(-2, 1, 2)
destination = np.matrix(destination)

proj = getPerspectiveTransform(source, destination)
output = warpPerspective(img, proj, (1000, 500))

cv2.imwrite('skewedOutput.jpg', output)

cv2.imshow("hello?", output)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
'''

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

cv2.imwrite("YESSSSSS.jpg", resultPicture)
