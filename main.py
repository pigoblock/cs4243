import os
import cv2
import cv2.cv as cv
import numpy as np
import matplotlib.pyplot as plt

global clickedList
global cornersList
global grayscalePicture
global colorPicture
global picHeight
global picWidth

# Store clicked coordinates
def onMouse(event, x, y, flags, param):
    if event == cv.CV_EVENT_LBUTTONDOWN:
        drawCircle(x, y, colorPicture, (0, 0, 255, 0))
        cv2.imshow("Picture", colorPicture)
        clickedList.append([x, y])
        getCalculatedGoodCorners(x, y)
        print "clicked at ", x, " , ", y

# Get and store recalculated corners based on what user clicked
def getCalculatedGoodCorners(clickedX, clickedY):    
    areaSize = 16
    extractedWindow = getExtractedArea(areaSize/2, clickedX, clickedY)
    harris = cv2.cornerHarris(extractedWindow, 2, 3, 0.04)

    # Get best corner
    cornerCoordinates = (0, 0)
    bestCornerValue = harris[0, 0]
    for x in range(areaSize):
        for y in range(areaSize):
            if harris[y, x] > bestCornerValue:
                bestCornerValue = harris[y, x]
                coordX = clickedX-areaSize/2+x
                coordY = clickedY-areaSize/2+y

    drawCircle(coordX, coordY, colorPicture, (0, 255, 0, 0))
    cornersList.append([coordX, coordY])
    print "recalculated corner: ", cornerCoordinates

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

clickedList = []
cornersList = []

grayscalePicture = cv2.imread("assets/project.jpeg", cv2.CV_LOAD_IMAGE_GRAYSCALE)
colorPicture = cv2.imread("assets/project.jpeg", cv2.CV_LOAD_IMAGE_COLOR)
picHeight= grayscalePicture.shape[0]
picWidth = grayscalePicture.shape[1]

cv2.namedWindow('Picture', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Picture', onMouse, 0)
cv2.imshow("Picture", colorPicture)
cv2.waitKey()

print cornersList
