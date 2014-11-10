import os
import cv2
import cv2.cv as cv
import numpy as np
import matplotlib.pyplot as plt

global faceList2D
global faceList3D
faceList2D = []
faceList3D = []

# 2D face object that stores the corner points, rotation and distance from camera of the face of a plane
class face2D:
    angleXZ = 0
    angleXY = 0
    distanceFromCamera = 0
    pointList = []

    def __init__(self, angleXZ, angleXY, distance):
        self.pointList = []
        self.angleXY = angleXY
        self.angleXZ = angleXZ
        self.distanceFromCamera = distance

# 2D point class for storing of 2D points
class point2D:
    x = 0
    y = 0

    def __init__(self, x, y):
        self.x = x
        self.y = y

# 3D point class for storing of 3D points. Inherits point2D class
class point3D(point2D):
    z = 0

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

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
    scale_factor = 0.9
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
            point1 = point3D(int(face.pointList[3].x), int(face.pointList[3].y), int(face.distanceFromCamera))
            point2 = point3D(int(face.pointList[2].x), int(face.pointList[2].y), int(face.distanceFromCamera))
            point3 = point3D(int(face.pointList[1].x), int(face.pointList[1].y), int(face.distanceFromCamera))
            point4 = point3D(int(face.pointList[0].x), int(face.pointList[0].y), int(face.distanceFromCamera))

            # Transforming points to world coordinates
            point1.x = int((point1.x - center_of_projection_x) * point1.z / focal_length * scale_factor)
            point1.y = int((point1.y - center_of_projection_y) * point1.z / focal_length * scale_factor)

            point2.x = int((point2.x - center_of_projection_x) * point2.z / focal_length * scale_factor)
            point2.y = int((point2.y - center_of_projection_y) * point2.z / focal_length * scale_factor)

            point3.x = int((point3.x - center_of_projection_x) * point3.z / focal_length * scale_factor)
            point3.y = int((point3.y - center_of_projection_y) * point3.z / focal_length * scale_factor)

            point4.x = int((point4.x - center_of_projection_x) * point4.z / focal_length * scale_factor)
            point4.y = int((point4.y - center_of_projection_y) * point4.z / focal_length * scale_factor)

            # Generating all corner points (assumed width of 100) of left and right face
            #point5 = point3D(point1.x, point1.y, int(point1.z) + point2.x - point1.x)
            #point6 = point3D(point2.x, point2.y, int(point2.z) + point2.x - point1.x)
            point7 = point3D(point3.x, point3.y, int(point3.z) + point2.x - point1.x)
            point8 = point3D(point4.x, point4.y, int(point4.z) + point2.x - point1.x)

            # Generating all possible points
            pointList3D = []
            # Generating all possible points of center face
            print "Generating 3d scene points for middle wall"
            for i in range(point1.x, point3.x):
                for j in range(point1.y, point3.y):
                    pointList3D.append(point3D(i, j, point1.z))
                    print "Appended point: " + str(pointList3D[len(pointList3D)-1].x) + ", " + str(pointList3D[len(pointList3D)-1].y) + ", " + str(pointList3D[len(pointList3D)-1].z)
            faceList3D.append(pointList3D)
            # Generating all possible points of left face
            print "Generating 3d scene points for left wall"
            for i in range(point1.z, point8.z):
                for j in range(point1.y, point8.y):
                    pointList3D.append(point3D(point1.x, j, i))
                    print "Appended point: " + str(pointList3D[len(pointList3D)-1].x) + ", " + str(pointList3D[len(pointList3D)-1].y) + ", " + str(pointList3D[len(pointList3D)-1].z)
            faceList3D.append(pointList3D)
            # Generating all possible points of right face
            print "Generating 3d scene points for right wall"
            for i in range(point2.z, point7.z):
                for j in range(point2.y, point7.y):
                    pointList3D.append(point3D(point2.x, j, i))
                    print "Appended point: " + str(pointList3D[len(pointList3D)-1].x) + ", " + str(pointList3D[len(pointList3D)-1].y) + ", " + str(pointList3D[len(pointList3D)-1].z)
            faceList3D.append(pointList3D)

        # point list corresponds to a wall facing to the right
        if int(face.angleXY) == 0:
            print "Generating 3d scene points for right wall"
            # Getting camera coordinates of points
            point1 = point3D(int(face.pointList[3].x), int(face.pointList[3].y), int(face.distanceFromCamera))
            point4 = point3D(int(face.pointList[0].x), int(face.pointList[0].y), int(face.distanceFromCamera))

            # Transforming points to world coordinates
            point1.x = int((point1.x - center_of_projection_x) * point1.z / focal_length * scale_factor)
            point1.y = int((point1.y - center_of_projection_y) * point1.z / focal_length * scale_factor)

            point4.x = int((point4.x - center_of_projection_x) * point4.z / focal_length * scale_factor)
            point4.y = int((point4.y - center_of_projection_y) * point4.z / focal_length * scale_factor)

            point2 = point3D(int(point1.x), int(point1.y), int(point1.z + point4.y - point1.y)) # point2 is behind point1
            point3 = point3D(int(point4.x), int(point4.y), int(point1.z + point4.y - point1.y)) # point3 is behind point4

            #print "Point 1: (" + str(point1.x) + ", " + str(point1.y) + ", " + str(point1.z) + ")"
            #print "Point 2: (" + str(point2.x) + ", " + str(point2.y) + ", " + str(point2.z) + ")"
            #print "Point 3: (" + str(point3.x) + ", " + str(point3.y) + ", " + str(point3.z) + ")"
            #print "Point 4: (" + str(point4.x) + ", " + str(point4.y) + ", " + str(point4.z) + ")"

            # Generating all possible points
            pointList3D = []
            for i in range(point1.z, point3.z):
                for j in range(point1.y, point3.y):
                    pointList3D.append(point3D(point1.x, j, i))
                    print "Appended point: " + str(pointList3D[len(pointList3D)-1].x) + ", " + str(pointList3D[len(pointList3D)-1].y) + ", " + str(pointList3D[len(pointList3D)-1].z)
            faceList3D.append(pointList3D)

        # point list corresponds to a wall facing to the left
        if int(face.angleXY) == 180:
            print "Generating 3d scene points for left wall"
            point2 = point3D(int(face.pointList[2].x), int(face.pointList[2].y), int(face.distanceFromCamera))
            point3 = point3D(int(face.pointList[1].x), int(face.pointList[1].y), int(face.distanceFromCamera))

            # Transforming points to world coordinates
            point2.x = int((point2.x - center_of_projection_x) * point2.z / focal_length * scale_factor)
            point2.y = int((point2.y - center_of_projection_y) * point2.z / focal_length * scale_factor)

            point3.x = int((point3.x - center_of_projection_x) * point3.z / focal_length * scale_factor)
            point3.y = int((point3.y - center_of_projection_y) * point3.z / focal_length * scale_factor)

            point1 = point3D(int(point2.x), int(point2.y), int(point2.z + point3.y - point2.y)) # point1 is behind point2
            point4 = point3D(int(point3.x), int(point3.y), int(point2.z + point3.y - point2.y)) # point4 is behind point3

            #print "Point 1: (" + str(point1.x) + ", " + str(point1.y) + ", " + str(point1.z) + ")"
            #print "Point 2: (" + str(point2.x) + ", " + str(point2.y) + ", " + str(point2.z) + ")"
            #print "Point 3: (" + str(point3.x) + ", " + str(point3.y) + ", " + str(point3.z) + ")"
            #print "Point 4: (" + str(point4.x) + ", " + str(point4.y) + ", " + str(point4.z) + ")"

            # Generating all possible points
            pointList3D = []
            for i in range(point3.z, point1.z):
                for j in range(point1.y, point3.y):
                    pointList3D.append(point3D(point1.x, j, i))
                    print "Appended point: " + str(pointList3D[len(pointList3D)-1].x) + ", " + str(pointList3D[len(pointList3D)-1].y) + ", " + str(pointList3D[len(pointList3D)-1].z)
            faceList3D.append(pointList3D)

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
colorPicture = cv2.imread("assets/project.jpeg", cv2.CV_LOAD_IMAGE_COLOR)
picHeight= grayscalePicture.shape[0]
picWidth = grayscalePicture.shape[1]
resultImg = np.zeros((picWidth,picHeight))
zBuffer = np.zeros((picWidth,picHeight))
#cv2.imshow("FWE", resultImg)
print "Interface is dumb. After every click, MUST input y (yes) or n (no) or the relevant values in order not to hang the program.\n"
print "Click in an clockwise manner, starting from the top left most point."
cv2.namedWindow('Picture', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Picture', onMouse, 0)
cv2.imshow("Picture", colorPicture)

'''
# Initialise resultant picture to be red
resultPicture = np.zeros((picHeight, picWidth, 3), np.uint8)
resultPicture[:] = (0, 0, 255)

# Hardcoded certain point for testing
pointsMatrix = np.float32(getPointsArray(-816,-612,816,612,100))
# Setting camera parameters
camera = np.matrix([[100.0, 0, 0],[0,100,0],[0,0,1]])
# Getting projected points, (pointsMatrix, rotation vector, translation vector, camera, coefficients)
resultPoints = cv2.projectPoints(pointsMatrix, (0,0,0), (0,100,30), camera, 0)

# For each point, check if in bounds and print, else nothing. Note that out of bounds on picture will cause wrap around.
for x in range(0, len(resultPoints[0])):
    print str(x) + "/" + str(len(resultPoints[0]))
    if resultPoints[0][x][0][1]+612 < picHeight and resultPoints[0][x][0][0]+816 < picWidth and resultPoints[0][x][0][1]+612  > 0 and resultPoints[0][x][0][0]+816 > 0:
        resultPicture[resultPoints[0][x][0][1]+612][resultPoints[0][x][0][0]+816] = colorPicture[pointsMatrix[x][1]+612,pointsMatrix[x][0]+816]
# Resultant picture
cv2.imshow("qwe", resultPicture)
'''

cv2.waitKey()
create3DscenePoints()
#colourAll2DPoints(<<2D array of points[x,y]>>, <<6D array of 3d points:[x,y,z,r,g,b]>>)
