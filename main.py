import cv2
import numpy as np
import math

# Data structures used for initializing
img = cv2.imread("assets/project.jpeg", cv2.CV_LOAD_IMAGE_COLOR)
baseImage = cv2.imread("assets/project.jpeg", cv2.CV_LOAD_IMAGE_COLOR)
picHeight= img.shape[0]
picWidth = img.shape[1]
array_2d_points_raw = []
array_3d_points_raw = []
array_points_per_plane = []
array_plane_direction = []
array_plane_description = []
polygonImage = np.zeros((picHeight, picWidth, 3), np.uint8)
# Camera parameters
focal_length = 300.0
center_of_projection_x = 767
center_of_projection_y = 850

def main():
    createVideoSequence()

def createVideoSequence():
    file_name = "result_"
    file_extension = ".jpg"
    frame_count = 0
    for i in range(1):
        r_vector = (0,0,0)
        t_vector = (0,5,-i/2.0)

    initializeVariablesFromFiles()
    r_vector = (0, 0, 0)
    t_vector = (0, 5, 0)
    resultImg = processImage(r_vector, t_vector, frame_count)
    #generalBlender(resultImg, picWidth, picHeight)
    #fillGapsAbsolute(resultImg, picWidth, picHeight)
    #deNoise(resultImg)
    cv2.imwrite(file_name + str(frame_count) + file_extension, resultImg)

    '''
    for i in range(50):
        initializeVariablesFromFiles()
        r_vector = (0, -0.001*i, 0)
        t_vector = (i*0.1, 10, -i*0.15)
        resultImg = processImage(r_vector, t_vector, frame_count)
        cv2.imwrite(file_name + str(frame_count) + file_extension, resultImg)
        frame_count += 1

    frame_count = 50
    for i in range(50):
        initializeVariablesFromFiles()
        r_vector = (0, -0.001*49 + 0.001*i, 0)
        t_vector = (49*0.1 - i*0.2, 10, -49*0.15 - i*0.1)
        resultImg = processImage(r_vector, t_vector, frame_count)
        cv2.imwrite(file_name + str(frame_count) + file_extension, resultImg)
        frame_count += 1

    frame_count = 100
    for i in range(50):
        initializeVariablesFromFiles()
        r_vector = (i*0.001, -0.001*49 + 0.001*49 - i*0.001, 0)
        t_vector = (49*0.1 - 49*0.2 + i*0.2, 10 + i*0.1, -49*0.15 - 49*0.1 - i*0.1)
        resultImg = processImage(r_vector, t_vector, frame_count)
        cv2.imwrite(file_name + str(frame_count) + file_extension, resultImg)
        frame_count += 1
    '''

# Initialize arrays given 2 input file obtained using InputInterface.py
# points.txt will be sliced into 2 arrays: one containing 2d points, and the other containing the corresponding 3d points
# planeDetails.txt will be sliced into 2 arrays: one containing number of points in one plane, and the other containing the plane direction
def initializeVariablesFromFiles():
    # Parsing points.txt to initialize 2d and 3d points array
    #print "Parsing points.txt ..."
    pointFile = open('points.txt')
    temp_array = pointFile.read().splitlines()
    for i in range(len(temp_array)):
        temp_line = temp_array[i].split()
        if len(temp_line) == 0:
            continue
        else:
            array_2d_points_raw.append([int(temp_line[0]), int(temp_line[1])])
            array_3d_points_raw.append([int(temp_line[2]), int(temp_line[3]), int(temp_line[4])])
    pointFile.close()

    # Parsing planeDetail.txt to initialize plane detail arrays
    #print "Parsing planeDetails.txt ..."
    planeDetails = open('planeDetails.txt')
    temp_array = planeDetails.read().splitlines()
    for i in range(len(temp_array)):
        temp_line = temp_array[i].split()
        if len(temp_line) == 0:
            continue
        else:
            array_points_per_plane.append(int(temp_line[0]))
            array_plane_direction.append(str(temp_line[1]))
            array_plane_description.append(str(temp_line[2]))
    planeDetails.close()

# For all planes, perform homography
def processImage(r_vector, t_vector, sequence_number):
    print "Processing sequence " + str(sequence_number)
    #resultImg = np.zeros((picHeight ,picWidth, 3))

    skyImg = cv2.imread("sky.jpg", cv2.CV_LOAD_IMAGE_COLOR)
    resultImg = skyImg

    num_planes = len(array_points_per_plane)
    # For all planes
    for i in range(num_planes):
        # Obtain plane detail for i-th plane
        numPoints = array_points_per_plane[i]
        planeDirection = array_plane_direction[i]
        planeDescription = array_plane_description[i]
        if i == 0:
            isGrass = True
        else:
            isGrass = False

        resultImg = performHomography(numPoints, planeDirection, resultImg, r_vector, t_vector, planeDescription, isGrass)
    return resultImg

# Given plane details, function will generate all possible 3d points and attach pixel color found through warped transformation
def performHomography(numPoints, planeDirection, resultImg, r_vector, t_vector, planeDescription, isGrass):
    #print "Performing homography for plane: " + planeDescription
    array_points_2d = array_2d_points_raw[:numPoints]
    array_points_3d = array_3d_points_raw[:numPoints]
    del array_2d_points_raw[:numPoints]
    del array_3d_points_raw[:numPoints]

    if len(array_points_2d) == 0 or len(array_points_3d) == 0:
        return resultImg

    if isGrass == True:
        array_points_3d[2][2] = t_vector[2] *-1
        array_points_3d[3][2] = t_vector[2] *-1

    xStart = array_points_2d[0][0]
    xEnd = array_points_2d[1][0]
    yStart = array_points_2d[0][1]
    yEnd = array_points_2d[3][1]
    tempImg = cropImage(baseImage, xStart, xEnd, yStart, yEnd)

    array_points = getCorners(tempImg)
    print array_points
    camera = np.matrix([[focal_length, 0, center_of_projection_x],[0, focal_length, center_of_projection_y],[0, 0, 1]])
    resultPoints = getProjectedPoints(np.array(array_points_3d, np.float32), r_vector, t_vector, camera, 0)
    warppedImage = warpHomo(np.array(array_points, np.float32), np.array(resultPoints, np.float32), (picWidth, picHeight ), tempImg)

    #duplicateImage = resultImg.copy()
    resultImg = overlayImage(resultImg, warppedImage)
    return resultImg

#Gets image pixels from xStart to xEnd -1, yStart to yEnd - 1
def cropImage(image, xStart, xEnd, yStart, yEnd):
    cropped = image[yStart:yEnd, xStart:xEnd]
    return cropped

#Gets Corners of 2D image in clockwise direction
def getCorners(image):
    return [[0,0],[len(image[0]), 0], [len(image[0]), len(image)], [0, len(image)]]

#Gets 4 projected points of a rectangle in 3D space
def getProjectedPoints(points_3D, Rvect, Tvect, camera, distVect):
    resultPoints = cv2.projectPoints(points_3D, Rvect, Tvect, camera, distVect)
    resultPoints2 = [[resultPoints[0][0][0][0], resultPoints[0][0][0][1]],
                 [resultPoints[0][1][0][0], resultPoints[0][1][0][1]],
                 [resultPoints[0][2][0][0], resultPoints[0][2][0][1]],
                 [resultPoints[0][3][0][0], resultPoints[0][3][0][1]]]
    return resultPoints2

#Points should be in np.array float32 format, Returns the warped picture with black BG on ouputImgSize dimensions
def warpHomo(imgPts_2D, projPts_2D, outputImgSize, croppedImg):
    H = cv2.findHomography(imgPts_2D, projPts_2D , 0 );
    homoImage = np.zeros((picHeight,  picWidth, 3), np.uint8)
    homoImage[:] = (0, 0, 255)
    cv2.warpPerspective(croppedImg, H[0], outputImgSize, homoImage)
    return homoImage

def overlayImage(resultImage, homoImage, resultPoints, planeDirection):
    
    hasProblems = False
    problemCase = 0
    
    point1 = resultPoints[0]
    point2 = resultPoints[1]
    point3 = resultPoints[2]
    point4 = resultPoints[3]
    print resultPoints

    if (point1[1] > point4[1]) :
        print "Failed case 1"
        hasProblems = True
        problemCase += 1

    if (point2[1] > point3[1]) :
        print "Failed case 2"
        hasProblems = True
        problemCase += 2

    if (point1[0] < point2[0] and point3[0] < point4[0]):
        print "Failed case 3"
        hasProblems = True
        problemCase += 4

    if (hasProblems):
        print problemCase
        cv2.imwrite("error.jpg", homoImage)
    
    greyHomo = cv2.cvtColor(homoImage, cv2.cv.CV_BGR2GRAY)
    retVal, greyHomo2 = cv2.threshold(greyHomo, 0,255,cv2.cv.CV_THRESH_BINARY)
    if (hasProblems) :
        if (problemCase >= 4) :
            yMin = point1[1]
            if (point2[1] < yMin):
                yMin = point2[1]
            greyHomo2[:yMin] = (0)
        if (planeDirection == 'up'):
            yMin = point1[1]
            if (point2[1] < yMin):
                yMin = point2[1]
            greyHomo2[:yMin] = (0)
        if (planeDirection == 'left'):
            xMin = point1[0]
            if (point4[0] < xMin):
                xMin = point4[0]
            greyHomo2[:,0:xMin] = 0
        if (planeDirection == 'right'):
            xMax = point2[0]
            if (point3[0] < xMax):
                xMax = point3[0]
            greyHomo2[:,xMax:] = 0
        cv2.imwrite("error.jpg", greyHomo2)
       # cv2.imshow("qweqwew", greyHomo2)
        #cv2.waitKey()
        
    greyHomo_inv = ~greyHomo2
    newResultImage = np.zeros((picHeight,  picWidth, 3), np.uint8)
    newResultImage[:] = (0, 0, 0)
    newHomoImage = np.zeros(( picHeight, picWidth , 3), np.uint8)
    newHomoImage[:] = (0, 0, 0)

    cv2.bitwise_and(resultImage, resultImage, newResultImage,greyHomo_inv)
    cv2.bitwise_and(homoImage, homoImage, newHomoImage,greyHomo2)

    resultImage = newResultImage + newHomoImage
    return resultImage

# Extract and returns an area from picture
def getExtractedArea(pic, areaRadius, centerX, centerY):
    extractedArea = pic[:, centerX-areaRadius:centerX+areaRadius]
    extractedArea = extractedArea[centerY-areaRadius:centerY+areaRadius, :]

    return extractedArea

# Fills in black colored holes with a selected color of pixel of furthest color distance
def fillGapsAbsolute(picture, width, height):
    hsvPic = cv2.cvtColor(picture, cv2.COLOR_BGR2HSV)
    for x in range (width):
        for y in range (height):          
            intensity = picture[y, x]
            blue = int(intensity[0])
            green = int(intensity[1])
            red = int(intensity[2])

            hsv = hsvPic[y, x]
            h = int(hsv[0])
            s = int(hsv[1])
            v = int(hsv[2])
            
            # if it is a hole
            if (v <= 30):
                #print "Hole found at [", x, ", ", y, "]"
                surroundingPixels = getExtractedArea(picture, 3, x, y)
                spHeight = surroundingPixels.shape[0]
                spWidth = surroundingPixels.shape[1]

                maxColorDistance = 0
                maxR = 0
                maxG = 0
                maxB = 0
                for i in range (spWidth):
                    for j in range (spHeight):
                        spIntensity = surroundingPixels[j, i]
                        totalIntensity = (int(spIntensity[0])-blue)**2 + (int(spIntensity[1])-green)**2 + (int(spIntensity[2])-red)**2
                        # Get color of max distance
                        if (totalIntensity > maxColorDistance):
                            maxColorDistance = totalIntensity
                            maxB = spIntensity[0]
                            maxG = spIntensity[1]
                            maxR = spIntensity[2]

                #Fill in the hole
                picture[y, x] = (maxB, maxG, maxR)

# Takes the average colour of surrounding pixels to paint black pixel holes 
def generalBlender(picture, width, height):
    hsvPic = cv2.cvtColor(picture, cv2.COLOR_BGR2HSV)
    for x in range (width):
        for y in range (height):
            intensity = picture[y, x];
            blue = intensity[0];
            green = intensity[1];
            red = intensity[2];

            avgBlue = 0;
            avgGreen = 0;
            avgRed = 0;
            numValidSP = 0;

            hsv = hsvPic[y, x]
            h = int(hsv[0])
            s = int(hsv[1])
            v = int(hsv[2])

            # if it is a hole
            if (v <= 30):
                #print "Hole found at [", x, ", ", y, "]"
                surroundingPixels = getExtractedArea(picture, 3, x, y)
                spHeight = surroundingPixels.shape[0]
                spWidth = surroundingPixels.shape[1]
                for i in range (spWidth):
                    for j in range (spHeight):
                        spIntensity = surroundingPixels[j, i];
                        if (i != x and j != y and spIntensity[0] > 50 and spIntensity[1] > 50 and spIntensity[2] > 50):
                            avgBlue += spIntensity[0]
                            avgGreen += spIntensity[1]
                            avgRed += spIntensity[2]
                            numValidSP += 1
                if (numValidSP > 0):
                    avgBlue /= numValidSP
                    avgGreen /= numValidSP
                    avgRed /= numValidSP

                #Fill in the hole
                picture[y, x] = (avgBlue, avgGreen, avgRed)

def deNoise(image):
    cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21) 
    
main()
