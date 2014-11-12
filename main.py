import os
import cv2
import cv2.cv as cv
import numpy as np
import matplotlib.pyplot as plt
from cv2 import getPerspectiveTransform, warpPerspective, imshow

# Data structures used for initializing
img = cv2.imread("assets/project.jpeg", cv2.CV_LOAD_IMAGE_COLOR)
array_2d_points_raw = []
array_3d_points_raw = []
array_points_per_plane = []
array_plane_direction = []
# Data structures obtained after processing (used for the rest of the whole project)
array_3d_points_with_color = []
sceneSize = []

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

    scene_width, scene_height = getSceneSize(array_points_3d, planeDirection)

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
                # point 1 = top left corner of plane
                # point 1 add scene width in the x direction
                # point 1 add scene height in the y direction
                # point 1 = output[j][i]
                pixel_color = output[j][i]
                array_3d_points_with_color.append([point1[0] + i, point1[1] + j, point1[2], pixel_color[0], pixel_color[1], pixel_color[2]])
            if planeDirection == 'left':
                # point 1 = top left corner of plane
                # point 1 add scene width in the z direction
                # point 1 add scene height in the y direction
                # point 1 = output[j][i]
                array_3d_points_with_color.append([1,2,3,4,5,6])
            if planeDirection == 'right':
                # point 1 = top left corner of plane
                # point 1 add scene width in the -z direction
                # point 1 add scene height in the y direction
                # point 1 = output[j][i]
                array_3d_points_with_color.append([1,2,3,4,5,6])
            if planeDirection == 'up':
                point1 = array_points_3d[0]
                # point 1 = top left corner of plane
                # point 1 add scene width in the x direction
                # point 1 add scene height in the -z direction
                # point 1 = output[j][i]
                pixel_color = output[j][i]
                array_3d_points_with_color.append([point1[0] + i, point1[1], point1[2] - j, pixel_color[0], pixel_color[1], pixel_color[2]])
                #print "Appended point: " + str(array_3d_points_with_color[len(array_3d_points_with_color)-1])
            if planeDirection == 'down':
                # point 1 = top left corner of plane
                # point 1 add scene width in the x direction
                # point 1 add scene height in the z direction
                # point 1 = output[j][i]
                array_3d_points_with_color.append([1,2,3,4,5,6])

# Given 4 points and where the plane is facing, function returns plane width and height
def getSceneSize(points, planeDirection):

    if(planeDirection == 'front'):
        # width = point3.x - point1.x
        sceneWidth = abs(points[2][0] - points[0][0])
        # height = point3.y - point1.y
        sceneHeight = abs(points[2][1] - points[0][1])
        return sceneWidth, sceneHeight
    if(planeDirection == 'left'):
        return 500, 500
    if(planeDirection == 'right'):
        return 500, 500
    if(planeDirection == 'up'):
        # width = point3.x - point1.x
        sceneWidth = abs(points[2][0] - points[0][0])
        # height = point3.z - point1.z
        sceneHeight = abs(points[2][2] - points[0][2])
        return sceneWidth, sceneHeight
    if(planeDirection == 'down'):
        return 1000, 500


global faceList2D
global faceList3D
faceList2D = []
faceList3D = []

# Mouse click event that stores all values that will be needed for further calculation
def onMouse(event, x, y, flags, param):
    global numPlanes
    if event == cv.CV_EVENT_LBUTTONDOWN:
        drawCircle(x, y, colorPicture, (0, 0, 255, 0))
        cv2.imshow("Picture", colorPicture)
        clickedList.append([numPlanes, x, y])
        print "clicked at ", x, " ,", y
        tempList.append(point2D(x, y))

        getCalculatedGoodCorners(x, y)
        userInput = raw_input("Done selecting corners for a plane? y/n ")
        if userInput == "y":
            numPlanes = numPlanes + 1
            angleInput = raw_input("Angle wrt rotation of xz plane (aka ground = 0, wall = 90, etc) :  ")
            faceInput = raw_input("Angle wrt rotation of xy plane (aka face right = 0, face left = 180, etc.) :  ")
            distanceInput = raw_input("Distance of plane (estimate how far the plane is from our view) :  ")
            angleList.append([angleInput, faceInput, distanceInput])

            # Creates face object
            angleXZ = angleInput
            angleXY = faceInput
            plane = face2D(angleXZ, angleXY, distanceInput)
            while len(tempList) != 0:
                plane.pointList.append(tempList.pop())
            faceList2D.append(plane)

            userInput = raw_input("Done selecting all planes? y/n ")
            print "\n"
            if userInput == "y":
                cv2.destroyWindow("Picture")
            else:
                return
        else:
            return

# Get and store recalculated corners based on what user clicked
def getCalculatedGoodCorners(clickedX, clickedY):
    areaSize = 10
    extractedWindow = getExtractedArea(areaSize/2, clickedX, clickedY)
    harris = cv2.cornerHarris(extractedWindow, 2, 3, 0.04)

    # Get best corner
    cornerCoordinates = (0, 0)
    bestCornerValue = harris[0, 0]
    for x in range(areaSize):
        for y in range(areaSize):
            if harris[y, x] > bestCornerValue:
                bestCornerValue = harris[y, x]
                coordX = clickedX - areaSize/2+x
                coordY = clickedY - areaSize/2+y

    drawCircle(coordX, coordY, colorPicture, (0, 255, 0, 0))
    cornersList.append([numPlanes, coordX, coordY])
    print "recalculated corner: ", coordX, " ,", coordY

# Extract and returns an area from picture
def getExtractedArea(areaRadius, centerX, centerY):
    extractedArea = grayscalePicture[:, centerX-areaRadius:centerX+areaRadius]
    extractedArea = extractedArea[centerY-areaRadius:centerY+areaRadius, :]
    return extractedArea

def drawCircle(x, y, image, color):
    center = (x,y)
    radius = 5
    thickness = -1 # negative means filled
    lineType = 8
    shift = 0
    cv2.circle(image, center, radius, color, thickness, lineType, shift)

def getPointsArray(x1, y1, x2, y2, distance):
    points = []
    for x in range(x1, x2-1):
        for y in range(y1, y2):
            print x,
            print y
            points.append([x,y,distance])
    return points

def debugFaceList2D():
    for i in range(len(faceList2D)):
        print "Face number: " + str(i)
        print "Angle w.r.t. ground (plane XZ): " + faceList2D[i].angleXZ
        print "Angle w.r.t. direction (plane XY): " + faceList2D[i].angleXY
        print "Distance from camera: " + faceList2D[i].distanceFromCamera
        print "Point List:"
        for j in range(len(faceList2D[i].pointList)):
            print faceList2D[i].pointList[j].x,
            print faceList2D[i].pointList[j].y

# Create 3D scene points using faceList2D
# Camera Position assumed to be at (?, ?, ?)
# Focal length assumed to be 1
# Points assumed to be clicked in a clockwise manner, starting from top left
def create3DscenePoints():
    for i in range(len(faceList2D)):
        numPoints = len(faceList2D[i].pointList)
        if(numPoints == 4):
            generate3DscenePointsRectangle(faceList2D[i])
        if(numPoints == 3):
            generate3DscenePointsTriangle(faceList2D[i])
    print "Total face generated in 3D points:" + str(len(faceList3D))

def generate3DscenePointsRectangle(face):
    focal_length = 30.0
    scale_factor = 1
    center_of_projection_x = 816.0
    center_of_projection_y = 612.0
    print "Generating 3d scene points for rectangle"
    # Point list corresponds to a ground
    if int(face.angleXZ) == 0:
        #generate points for ground
        print "Generating 3d scene points for ground"
    # Point list corresponds to a wall
    if int(face.angleXZ) == 90:
        # point list corresponds to a wall facing the center
        if int(face.angleXY) == 90:
            point1 = [int(face.pointList[3].x), int(face.pointList[3].y), int(face.distanceFromCamera)]
            point2 = [int(face.pointList[2].x), int(face.pointList[2].y), int(face.distanceFromCamera)]
            point3 = [int(face.pointList[1].x), int(face.pointList[1].y), int(face.distanceFromCamera)]
            point4 = [int(face.pointList[0].x), int(face.pointList[0].y), int(face.distanceFromCamera)]

            # Transforming points to world coordinates
            point1[0] = int((point1[0] - center_of_projection_x) * point1[2] / focal_length * scale_factor)
            point1[1] = int((point1[1] - center_of_projection_y) * point1[2] / focal_length * scale_factor)

            point2[0] = int((point2[0] - center_of_projection_x) * point2[2] / focal_length * scale_factor)
            point2[1] = int((point2[1] - center_of_projection_y) * point2[2] / focal_length * scale_factor)

            point3[0] = int((point3[0] - center_of_projection_x) * point3[2] / focal_length * scale_factor)
            point3[1] = int((point3[1] - center_of_projection_y) * point3[2] / focal_length * scale_factor)

            point4[0] = int((point4[0] - center_of_projection_x) * point4[2] / focal_length * scale_factor)
            point4[1] = int((point4[1] - center_of_projection_y) * point4[2] / focal_length * scale_factor)

            # Generating all corner points (assumed width of 100) of left and right face
            point5 = [point1[0], point1[1], point1[2] + point2[0] - point1[0]]
            point6 = [point2[0], point2[1], point2[2] + point2[0] - point1[0]]
            point7 = [point3[0], point3[1], point3[2] + point2[0] - point1[0]]
            point8 = [point4[0], point4[1], point4[2] + point2[0] - point1[0]]

            # Generating all possible points of center face
            print "Generating 3d scene points for middle wall"
            for i in range(point1[0], point3[0]):
                for j in range(point1[1], point3[1]):
                    # backtrack to 2D image to obtain colour for the 3D point
                    camera_position_x = int(i * focal_length / point1[2] / scale_factor + center_of_projection_x)
                    camera_position_y = int(j * focal_length / point1[2] / scale_factor + center_of_projection_y)
                    pixel_color = baseImage[camera_position_y, camera_position_x]
                    print "Taking pixel color of coordinate: [" + str(camera_position_x) + ", " + str(camera_position_y) + "]"
                    # Append 3D point (x, y, z, blue intensity, green intensity, red intensity)
                    faceList3D.append([i, j, point1[2], pixel_color[0], pixel_color[1], pixel_color[2]])
                    print "Appended point for middle wall: ",
                    print faceList3D[len(faceList3D)-1]

            # Generating all possible points of left face
            print "Generating 3d scene points for left wall"
            for i in range(point1[1], point8[1]):
                # Get pixel color of front plane
                camera_position_x = int(point1[0] * focal_length / point1[2] / scale_factor + center_of_projection_x)
                camera_position_y = int(i * focal_length / point1[2] / scale_factor + center_of_projection_y)
                pixel_color = baseImage[camera_position_y, camera_position_x]
                for j in range(point1[2], point8[2]):
                    print "Taking pixel color of coordinate: [" + str(camera_position_x) + ", " + str(camera_position_y) + "]"
                    # Append 3D point (x, y, z, blue intensity, green intensity, red intensity)
                    faceList3D.append([point1[0], i, j, pixel_color[0], pixel_color[1], pixel_color[2]])
                    print "Appended point for left wall: ",
                    print faceList3D[len(faceList3D)-1]

            # Generating all possible points of right face
            print "Generating 3d scene points for right wall"
            for i in range(point2[1], point7[1]):
                camera_position_x = int(point2[0] * focal_length / point1[2] / scale_factor + center_of_projection_x)
                camera_position_y = int(i * focal_length / point1[2] / scale_factor + center_of_projection_y)
                pixel_color = baseImage[camera_position_y, camera_position_x]
                for j in range(point2[2], point7[2]):
                    print "Taking pixel color of coordinate: [" + str(camera_position_x) + ", " + str(camera_position_y) + "]"
                    # Append 3D point (x, y, z, blue intensity, green intensity, red intensity)
                    faceList3D.append([point2[0], i, j, pixel_color[0], pixel_color[1], pixel_color[2]])
                    print "Appended point for right wall: ",
                    print faceList3D[len(faceList3D)-1]

                print point1
                print point2
                print point3
                print point4

        # point list corresponds to a wall facing to the right
        if int(face.angleXY) == 0:
            print "Generating 3d scene points for right wall"
            # Getting camera coordinates of points
            point1 = [int(face.pointList[3].x), int(face.pointList[3].y), int(face.distanceFromCamera)]
            point2 = [int(face.pointList[2].x), int(face.pointList[2].y), int(face.distanceFromCamera)]
            point3 = [int(face.pointList[1].x), int(face.pointList[1].y), int(face.distanceFromCamera)]
            point4 = [int(face.pointList[0].x), int(face.pointList[0].y), int(face.distanceFromCamera)]

            # Transforming points to world coordinates
            point1[0] = int((point1[0] - center_of_projection_x) * point1[2] / focal_length * scale_factor)
            point1[1] = int((point1[1] - center_of_projection_y) * point1[2] / focal_length * scale_factor)

            point4[0] = int((point4[0] - center_of_projection_x) * point4[2] / focal_length * scale_factor)
            point4[1] = int((point4[1] - center_of_projection_y) * point4[2] / focal_length * scale_factor)

            point2 = [int(point1[0]), int(point1[1]), int(point1[2] + point4[1] - point1[1])] # point2 is behind point1
            point3 = [int(point4[0]), int(point4[1]), int(point1[2] + point4[1] - point1[1])] # point3 is behind point4

            print point1
            print (point2[0] / point2[2] * focal_length / scale_factor) + center_of_projection_x,
            print (point2[1] / point2[2] * focal_length / scale_factor) + center_of_projection_y
            print (point3[0] / point3[2] * focal_length / scale_factor) + center_of_projection_x,
            print (point3[1] / point3[2] * focal_length / scale_factor) + center_of_projection_y
            print point4

            # Generating all possible points
            for i in range(point1[1], point3[1]):
                for j in range(point1[2], point3[2]):
                    camera_position_x = int(i * focal_length / point1[2] / scale_factor + center_of_projection_x)
                    camera_position_y = int(i * focal_length / point1[2] / scale_factor + center_of_projection_y)
                    pixel_color = baseImage[camera_position_x, camera_position_y]
                    faceList3D.append([point1[0], i, j])
                    print "Appended point for right wall: ",
                    print faceList3D[len(faceList3D)-1]

        # point list corresponds to a wall facing to the left
        if int(face.angleXY) == 180:
            print "Generating 3d scene points for left wall"
            point2 = [int(face.pointList[2].x), int(face.pointList[2].y), int(face.distanceFromCamera)]
            point3 = [int(face.pointList[1].x), int(face.pointList[1].y), int(face.distanceFromCamera)]

            # Transforming points to world coordinates
            point2[0] = int((point2[0] - center_of_projection_x) * point2[2] / focal_length * scale_factor)
            point2[1] = int((point2[1] - center_of_projection_y) * point2[2] / focal_length * scale_factor)

            point3[0] = int((point3[0] - center_of_projection_x) * point3[2] / focal_length * scale_factor)
            point3[1] = int((point3[1] - center_of_projection_y) * point3[2] / focal_length * scale_factor)

            point1 = [int(point2[0]), int(point2[1]), int(point2[2] + point3[1] - point2[1])] # point1 is behind point2
            point4 = [int(point3[0]), int(point3[1]), int(point2[2] + point3[1] - point2[1])] # point4 is behind point3

            print (point1[0] / point1[2] * focal_length / scale_factor) + center_of_projection_x,
            print (point1[1] / point1[2] * focal_length / scale_factor) + center_of_projection_y
            print (point2[0] / point2[2] * focal_length / scale_factor) + center_of_projection_x,
            print (point2[1] / point2[2] * focal_length / scale_factor) + center_of_projection_y
            print (point3[0] / point3[2] * focal_length / scale_factor) + center_of_projection_x,
            print (point3[1] / point3[2] * focal_length / scale_factor) + center_of_projection_y
            print (point4[0] / point4[2] * focal_length / scale_factor) + center_of_projection_x,
            print (point4[1] / point4[2] * focal_length / scale_factor) + center_of_projection_y

            '''
            source = np.array([[point1[0],point1[1]],[point2[0],point2[1]],[point3[0],point3[1]],[point4[0],point4[1]]],np.float32)
            source = np.array(source)

            destination = np.array([[0,0],[1000,0],[1000,800],[0,800]], np.float32)
            destination = destination.reshape(-2, 1, 2)
            destination = np.matrix(destination)

            proj = getPerspectiveTransform(source, destination)
            output = warpPerspective(colorPicture, proj, (1000,1000))
            cv2.imshow("hello?", output)
            if cv2.waitKey(0) == 27:
                cv2.destroyAllWindows()
            '''

            '''
            # Generating all possible points
            pointList3D = []
            for i in range(point3[2], point1[2]):
                for j in range(point1[1], point3[1]):
                    pointList3D.append([point1[0], j, i])
                    print "Appended point: ",
                    print pointList3D[len(pointList3D)-1]
            faceList3D.append(pointList3D)
            '''
def generate3DscenePointsTriangle():
    print "Generating 3d scene points for triangle"

def colour2Dpoint(point_2d, point_3d):
    if zBuffer[point_2d[0]][point_2d[1]] == 0 or point_3d[2] > zBuffer[point_2d[0]][point_2d[1]]:
        resultImg[point_2d[0]][point_2d[1]] = point_3d[3:]
        zBuffer[point_2d[0]][point_2d[1]] = point_3d[2]

def colourAll2DPoints(arr_2dPts, arr_3dPts):
    for x in range (0, len(arr_2dPts)):
        colour2Dpoint(arr_2dPts[x], arr_3dPts[x])

global clickedList
global cornersList
global angleList
global grayscalePicture
global colorPicture
global picHeight
global picWidth

global tempList
tempList = []

numPlanes = 0
clickedList = []
cornersList = []
angleList = []

grayscalePicture = cv2.imread("assets/project.jpeg", cv2.CV_LOAD_IMAGE_GRAYSCALE)
baseImage = cv2.imread("assets/project.jpeg", cv2.CV_LOAD_IMAGE_COLOR)
colorPicture = cv2.imread("assets/project.jpeg", cv2.CV_LOAD_IMAGE_COLOR)
picHeight= grayscalePicture.shape[0]
picWidth = grayscalePicture.shape[1]
resultImg = np.zeros((picWidth,picHeight))
zBuffer = np.zeros((picWidth,picHeight))
#cv2.imshow("FWE", resultImg)

initialize()
processInput()

'''
create3DscenePoints()
'''
# Initialise resultant picture to be red
resultPicture = np.zeros((picHeight, picWidth, 3), np.uint8)
resultPicture[:] = (0, 0, 0)

# Hardcoded certain point for testing
#pointsMatrix = np.float32(getPointsArray(-816,-612,816,612,100))
pointsMatrix = np.float32([array_3d_points_with_color[i][0:3] for i in range(0,len(array_3d_points_with_color))])
# Setting camera parameters
camera = np.matrix([[30.0, 0, 770],[0, 30.0, 860],[0, 0, 1]])
# Getting projected points, (pointsMatrix, rotation vector, translation vector, camera, coefficients)
resultPoints = cv2.projectPoints(pointsMatrix, (0, 0, 0), (0, 20.0, 0), camera, 0)

# For each point, check if in bounds and print, else nothing. Note that out of bounds on picture will cause wrap around.
for x in range(0, len(resultPoints[0])):
    print str(x) + "/" + str(len(resultPoints[0]))
    if resultPoints[0][x][0][1] < picHeight and resultPoints[0][x][0][0] < picWidth and resultPoints[0][x][0][1]  > 0 and resultPoints[0][x][0][0] > 0:
        #resultPicture[resultPoints[0][x][0][1]][resultPoints[0][x][0][0]] = colorPicture[pointsMatrix[x][1],pointsMatrix[x][0]]
        resultPicture[resultPoints[0][x][0][1]][resultPoints[0][x][0][0]] = [array_3d_points_with_color[x][3], array_3d_points_with_color[x][4], array_3d_points_with_color[x][5]]

'''
print resultPoints[0]
# For each point, check if in bounds and print, else nothing. Note that out of bounds on picture will cause wrap around.
for x in range(0, len(resultPoints[0])):
    print str(x) + "/" + str(len(resultPoints[0])),
    print resultPoints
    if resultPoints[0][x][0][1]+612 < picHeight and resultPoints[0][x][0][0]+816 < picWidth and resultPoints[0][x][0][1]+612  > 0 and resultPoints[0][x][0][0]+816 > 0:
        resultPicture[resultPoints[0][x][0][1]+612][resultPoints[0][x][0][0]+816] = colorPicture[pointsMatrix[x][1]+612,pointsMatrix[x][0]+816]
'''
# Resultant picture
cv2.imshow("qwe", resultPicture)
cv2.imwrite("result.jpg", resultPicture);
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
