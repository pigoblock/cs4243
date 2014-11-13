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

image = cv2.imread("assets/project.jpeg", cv2.CV_LOAD_IMAGE_COLOR)

array_2d = []
point_2d_1 = [1175.0, 699]
point_2d_2 = [1240.0, 682]
point_2d_3 = [1240.0, 888]
point_2d_4 = [1175.0, 875]
array_2d.append(point_2d_1)
array_2d.append(point_2d_2)
array_2d.append(point_2d_3)
array_2d.append(point_2d_4)
'''
array_3d = []
point_3d_1 = [50.0, -30, 40]
point_3d_2 = [50.0, -30, 30]
point_3d_3 = [50.0, 0, 30]
point_3d_4 = [50.0, 0, 40]
array_3d.append(point_3d_1)
array_3d.append(point_3d_2)
array_3d.append(point_3d_3)
array_3d.append(point_3d_4)

pointsMatrix = np.float32([array_3d])

camera = np.matrix([[300.0, 0, 767],[0, 300.0, 850],[0, 0, 1]])
resultPoints = cv2.projectPoints(pointsMatrix, (0,0,0), (0,2,0), camera, 0)

resultPoints2 = [[int(resultPoints[0][0][0][0]), int(resultPoints[0][0][0][1])],
                 [int(resultPoints[0][1][0][0]), int(resultPoints[0][1][0][1])],
                 [int(resultPoints[0][2][0][0]), int(resultPoints[0][2][0][1])],
                 [int(resultPoints[0][3][0][0]), int(resultPoints[0][3][0][1])]
                 ]

print resultPoints2
'''
resultPoints2 = [[0,0],
                 [100,0],
                 [100,100],
                 [0,100]]

H = findHomography(np.array(array_2d, np.float32), np.array(resultPoints2, np.float32) , 0 );

picHeight= image.shape[0]
picWidth = image.shape[1]

image = warpPerspective(np.array(array_2d, np.float32), H[0], (picWidth, picHeight))

cv2.imwrite("YESSSSSS.jpg", image)
