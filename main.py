import cv2
import cv2.cv as cv
import numpy as np
import matplotlib.pyplot as plt
from cv2 import getPerspectiveTransform, warpPerspective, imshow
import math

# Data structures used for initializing
img = cv2.imread("assets/project.jpeg", cv2.CV_LOAD_IMAGE_COLOR)
array_2d_points_raw = []
array_3d_points_raw = []
array_points_per_plane = []
array_plane_direction = []
# Data structures obtained after processing (used for the rest of the whole project)
array_3d_points_with_color = []
sceneSize = []

corner_points = []

def toRad(degree):
    return degree*math.pi/180

def getRMatrix(xDeg, yDeg, zDeg):
    xDeg = toRad(xDeg)
    yDeg = toRad(yDeg)
    zDeg = toRad(zDeg)
    output = np.array([[math.cos(zDeg)*math.cos(yDeg), math.cos(zDeg)*math.sin(yDeg)*math.sin(xDeg)-math.sin(zDeg)*math.cos(xDeg), math.cos(zDeg)*math.sin(yDeg)*math.cos(xDeg) + math.sin(zDeg)*math.sin(xDeg)],
                      [math.sin(zDeg)*math.cos(yDeg), math.sin(zDeg)*math.sin(yDeg)*math.sin(xDeg)+math.cos(zDeg)*math.cos(xDeg), math.sin(zDeg)*math.sin(yDeg)*math.cos(xDeg) - math.cos(zDeg)*math.sin(xDeg)],
                      [-math.sin(yDeg), math.cos(yDeg)*math.sin(xDeg), math.cos(yDeg)*math.cos(xDeg)]])
    return output
# Initialize arrays given 2 input file obtained using InputInterface.py
# points.txt will be sliced into 2 arrays: one containing 2d points, and the other containing the corresponding 3d points
# planeDetails.txt will be sliced into 2 arrays: one containing number of points in one plane, and the other containing the plane direction
# Post condition: 4 arrays: array_2d_points_raw, array_3d_points_raw, array_points_per_plane, array_plane_direction
def initialize():

    # Parsing points.txt to initialize 2d and 3d points array
    print "Parsing points.txt ..."
    pointFile = open('points.txt')
    temp_array = pointFile.read().splitlines()
    for i in range(len(temp_array)):
        temp_line = temp_array[i].split()
        point_2d = [int(temp_line[0]), int(temp_line[1])]
        point_3d = [int(temp_line[2]), int(temp_line[3]), int(temp_line[4])]
        array_2d_points_raw.append(point_2d)
        array_3d_points_raw.append(point_3d)
    pointFile.close()

    # Parsing planeDetail.txt to initialize plane detail arrays
    print "Parsing planeDetails.txt ..."
    planeDetails = open('planeDetails.txt')
    temp_array = planeDetails.read().splitlines()
    for i in range(len(temp_array)):
        temp_line = temp_array[i].split()
        points_per_plane = int(temp_line[0])
        plane_direction = str(temp_line[1])
        array_points_per_plane.append(points_per_plane)
        array_plane_direction.append(plane_direction)
    planeDetails.close()

# For each plane detail (points per plane & direction), generate all 3d points using array_3d_points
# Color of each 3d point will be initialized using array_2d_points
def processInput():

    print "Processing input data ..."
    # For all planes
    for i in range(len(array_points_per_plane)):
        # Obtain plane detail for i-th plane
        numPoints = array_points_per_plane[i]
        planeDirection = array_plane_direction[i]
        if numPoints == 4:
            generate3DpointsRectangle(numPoints, planeDirection)
        if numPoints == 3:
            # Do something else
            print "yolo!"

# Given plane details, function will generate all possible 3d points and attach pixel color found through warped transformation
def generate3DpointsRectangle(numPoints, planeDirection):

    print "Generating 3d points for rectangle ... "
    array_points_2d = array_2d_points_raw[:numPoints]
    array_points_3d = array_3d_points_raw[:numPoints]
    del array_2d_points_raw[:numPoints]
    del array_3d_points_raw[:numPoints]

    # Painting with rectangles
    pointsMatrix = np.float32([array_points_3d])
    camera = np.matrix([[300.0, 0, 767],[0, 300.0, 850],[0, 0, 1]])
    resultPoints = cv2.projectPoints(pointsMatrix, (0, 0, 0), (0, 2, 0), camera, 0)
    '''
    point1 = resultPoints[0][0][0]
    point2 = resultPoints[0][1][0]
    point3 = resultPoints[0][2][0]
    point4 = resultPoints[0][3][0]
    corner_points.append(resultPoints[0][0][0])
    corner_points.append(resultPoints[0][1][0])
    corner_points.append(resultPoints[0][2][0])
    corner_points.append(resultPoints[0][3][0])
    polygon = np.array([point1, point2, point3, point4 ], np.int32)
    if planeDirection == 'up':
        cv2.fillConvexPoly(resultPicture, polygon, [39,92,66])
    else:
        cv2.fillConvexPoly(resultPicture, polygon, [141,175,204])
    '''
    scene_width, scene_height = getSceneSize(array_points_3d, planeDirection)
    scene_width *= 20
    scene_height *= 20
    # Obtain a warped transformation of 2d image into size of 3d image
    source = np.array(array_points_2d, np.float32)

    destination = np.array([[0,0],[scene_width,0],[scene_width,scene_height],[0,scene_height]], np.float32)
    destination = destination.reshape(-2, 1, 2)
    destination = np.matrix(destination)

    proj = getPerspectiveTransform(source, destination)
    output = warpPerspective(img, proj, (scene_width, scene_height))

    # Copy warped transformation into 3d image (direct mapping to scene plane)
    # Creates all possible 3d points and append pixel color to 3d array
    for j in range(scene_height):
        for i in range(scene_width):
            # Assign pixel color of warped transformation to 3d point
            if planeDirection == 'front':
                point1 = array_points_3d[0]
                pixel_color = output[j][i]
                # point 1 = top left corner of plane
                # point 1 add scene width in the x direction
                # point 1 add scene height in the y direction
                array_3d_points_with_color.append([point1[0] + (i/20.0), point1[1] + (j/20.0), point1[2], pixel_color[0], pixel_color[1], pixel_color[2]])

            if planeDirection == 'left':
                point1 = array_points_3d[0]
                pixel_color = output[j][i]
                # point 1 = top left corner of plane
                # point 1 add scene width in the z direction
                # point 1 add scene height in the y direction
                array_3d_points_with_color.append([point1[0], point1[1] + (j/20.0), point1[2] - (i/20.0), pixel_color[0], pixel_color[1], pixel_color[2]])
                #print "Appended point: " + str(array_3d_points_with_color[len(array_3d_points_with_color)-1])
                '''
            if planeDirection == 'right':
                # point 1 = top left corner of plane
                # point 1 add scene width in the -z direction
                # point 1 add scene height in the y direction
                pixel_color = output[j][i]
                array_3d_points_with_color.append([1,2,3,4,5,6])
            if planeDirection == 'up':
                point1 = array_points_3d[0]
                # point 1 = top left corner of plane
                # point 1 add scene width in the x direction
                # point 1 add scene height in the -z direction
                pixel_color = output[j][i]
                for m in range(10):
                    dx = 0.1 * m
                    for n in range(10):
                        dz = 0.1 * n
                        array_3d_points_with_color.append([point1[0] + i + dx, point1[1], point1[2] - j - dz, pixel_color[0], pixel_color[1], pixel_color[2]])
                        #print "Appended point: " + str(array_3d_points_with_color[len(array_3d_points_with_color)-1])
            if planeDirection == 'down':
                # point 1 = top left corner of plane
                # point 1 add scene width in the x direction
                # point 1 add scene height in the z direction
                pixel_color = output[j][i]
                array_3d_points_with_color.append([1,2,3,4,5,6])
    '''
    # Generate side walls

# Given 4 points and where the plane is facing, function returns plane width and height
def getSceneSize(points, planeDirection):

    if(planeDirection == 'front'):
        # width = point3.x - point1.x
        sceneWidth = abs(points[2][0] - points[0][0])
        # height = point3.y - point1.y
        sceneHeight = abs(points[2][1] - points[0][1])
        return sceneWidth, sceneHeight
    if(planeDirection == 'left'):
        # width = point1.z - point3.z
        sceneWidth = abs(points[0][2] - points[2][2])
        # height = point3.y - point1.y
        sceneHeight = abs(points[2][1] - points[0][1])
        return sceneWidth, sceneHeight
    if(planeDirection == 'right'):
        # width = point3.z - point1.z
        sceneWidth = abs(points[2][2] - points[0][2])
        # height = point3.y - point1.y
        sceneHeight = abs(points[2][1] - points[0][1])
        return sceneWidth, sceneHeight
    if(planeDirection == 'up'):
        # width = point3.x - point1.x
        sceneWidth = abs(points[2][0] - points[0][0])
        # height = point1.z - point3.z
        sceneHeight = abs(points[0][2] - points[2][2])
        return sceneWidth, sceneHeight
    if(planeDirection == 'down'):
        # width = point3.x - point1.x
        sceneWidth = abs(points[2][0] - points[0][0])
        # height = point3.z - point1.z
        sceneHeight = abs(points[2][2] - points[0][2])
        return sceneWidth, sceneHeight

def colour2Dpoint(point_2d, point_3d):
    if zBuffer[point_2d[0]][point_2d[1]] == 0 or point_3d[2] > zBuffer[point_2d[0]][point_2d[1]]:
        resultImg[point_2d[0]][point_2d[1]] = point_3d[3:]
        zBuffer[point_2d[0]][point_2d[1]] = point_3d[2]

def colourAll2DPoints(arr_2dPts, arr_3dPts):
    for x in range (0, len(arr_2dPts)):
        colour2Dpoint(arr_2dPts[x], arr_3dPts[x])

global grayscalePicture
global colorPicture
global picHeight
global picWidth

grayscalePicture = cv2.imread("assets/project.jpeg", cv2.CV_LOAD_IMAGE_GRAYSCALE)
baseImage = cv2.imread("assets/project.jpeg", cv2.CV_LOAD_IMAGE_COLOR)
colorPicture = cv2.imread("assets/project.jpeg", cv2.CV_LOAD_IMAGE_COLOR)
picHeight= grayscalePicture.shape[0]
picWidth = grayscalePicture.shape[1]
resultImg = np.zeros((picWidth,picHeight))
zBuffer = np.zeros((picWidth,picHeight))
#cv2.imshow("FWE", resultImg)

# Initialise resultant picture to be red
resultPicture = np.zeros((picHeight, picWidth, 3), np.uint8)
resultPicture[:] = (0, 0, 0)

initialize()
processInput()

# Hardcoded certain point for testing
#pointsMatrix = np.float32(getPointsArray(-816,-612,816,612,100))
pointsMatrix = np.float32([array_3d_points_with_color[i][0:3] for i in range(0,len(array_3d_points_with_color))])
# Setting camera parameters
camera = np.matrix([[300.0, 0, 767],[0, 300.0, 850],[0, 0, 1]])
# Getting projected points, (pointsMatrix, rotation vector, translation vector, camera, coefficients)
resultPoints = cv2.projectPoints(pointsMatrix, (0, 0, 0), (0, 2, 0), camera, 0)

# For each point, check if in bounds and print, else nothing. Note that out of bounds on picture will cause wrap around.
for x in range(0, len(resultPoints[0])):
    if resultPoints[0][x][0][1] < picHeight and resultPoints[0][x][0][0] < picWidth and resultPoints[0][x][0][1]  > 0 and resultPoints[0][x][0][0] > 0:
        #resultPicture[resultPoints[0][x][0][1]][resultPoints[0][x][0][0]] = colorPicture[pointsMatrix[x][1],pointsMatrix[x][0]]
        resultPicture[int(resultPoints[0][x][0][1])][int(resultPoints[0][x][0][0])] = [array_3d_points_with_color[x][3], array_3d_points_with_color[x][4], array_3d_points_with_color[x][5]]

# Resultant picture
cv2.imshow("qwe", resultPicture)
cv2.imwrite("result.jpg", resultPicture);