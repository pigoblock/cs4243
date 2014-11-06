import os
import cv2
import cv2.cv as cv
import numpy as np
import matplotlib.pyplot as plt

# Mouse click event that stores all values that will be needed for further calculation
def onMouse(event, x, y, flags, param):
    global numPlanes
    if event == cv.CV_EVENT_LBUTTONDOWN:
        drawCircle(x, y, colorPicture, (0, 0, 255, 0))
        cv2.imshow("Picture", colorPicture)
        clickedList.append([numPlanes, x, y])
        print "clicked at ", x, " ,", y
        getCalculatedGoodCorners(x, y)
        userInput = raw_input("Done selecting corners for a plane? y/n ")
        if userInput == "y":
            numPlanes = numPlanes + 1
            angleInput = raw_input("Angle wrt rotation of xz plane (aka ground = 0, wall = 90, etc) :  ")
            faceInput = raw_input("Angle wrt rotation of xy plane (aka face right = 0, face left = 180, etc.) :  ")
            distanceInput = raw_input("Distance of plane (estimate how far the plane is from our view) :  ")
            angleList.append([angleInput, faceInput, distanceInput])
            
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
            points.append([x,y,distance])
    return points

global clickedList
global cornersList
global angleList
global grayscalePicture
global colorPicture
global picHeight
global picWidth
import numpy

numPlanes = 0
clickedList = []
cornersList = []
angleList = []

grayscalePicture = cv2.imread("assets/project.jpeg", cv2.CV_LOAD_IMAGE_GRAYSCALE)
colorPicture = cv2.imread("assets/project.jpeg", cv2.CV_LOAD_IMAGE_COLOR)
picHeight= grayscalePicture.shape[0]
picWidth = grayscalePicture.shape[1]

#print "Interface is dumb. After every click, MUST input y (yes) or n (no) or the relevant values in order not to hang the program.\n"
#print cornersList
#print angleList

# Initialise resultant picture to be red
resultPicture = np.zeros((picHeight, picWidth, 3), np.uint8)
resultPicture[:] = (0, 0, 255)

# Hardcoded certain point for testing
pointsMatrix = numpy.float32(getPointsArray(-816,-612,816,612,100))
# Setting camera parameters
camera = numpy.matrix([[100.0, 0, 0],[0,100,0],[0,0,1]])
# Getting projected points, (pointsMatrix, rotation vector, translation vector, camera, coefficients)
resultPoints = cv2.projectPoints(pointsMatrix, (0,0,0), (0,100,30), camera, 0)

# For each point, check if in bounds and print, else nothing. Note that out of bounds on picture will cause wrap around.
for x in range(0, len(resultPoints[0])):
    if resultPoints[0][x][0][1]+612 < picHeight and resultPoints[0][x][0][0]+816 < picWidth and resultPoints[0][x][0][1]+612  > 0 and resultPoints[0][x][0][0]+816 > 0:
        resultPicture[resultPoints[0][x][0][1]+612][resultPoints[0][x][0][0]+816] = colorPicture[pointsMatrix[x][1]+612,pointsMatrix[x][0]+816]

# Resultant picture
cv2.imshow("qwe", resultPicture)
cv2.waitKey()
