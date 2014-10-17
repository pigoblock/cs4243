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


global clickedList
global cornersList
global angleList
global grayscalePicture
global colorPicture
global picHeight
global picWidth

numPlanes = 0
clickedList = []
cornersList = []
angleList = []

grayscalePicture = cv2.imread("assets/project.jpeg", cv2.CV_LOAD_IMAGE_GRAYSCALE)
colorPicture = cv2.imread("assets/project.jpeg", cv2.CV_LOAD_IMAGE_COLOR)
picHeight= grayscalePicture.shape[0]
picWidth = grayscalePicture.shape[1]

print "Interface is dumb. After every click, MUST input y (yes) or n (no) or the relevant values in order not to hang the program.\n"

cv2.namedWindow('Picture', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Picture', onMouse, 0)
cv2.imshow("Picture", colorPicture)
cv2.waitKey()

print cornersList
print angleList
