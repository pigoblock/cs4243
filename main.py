import cv2
import numpy as np

# Data structures used for initializing
img = cv2.imread("assets/project.jpeg", cv2.CV_LOAD_IMAGE_COLOR)
baseImage = cv2.imread("assets/project.jpeg", cv2.CV_LOAD_IMAGE_COLOR)
picHeight= img.shape[0]
picWidth = img.shape[1]
array_2d_points_raw = []
array_3d_points_raw = []
array_points_per_plane = []
array_plane_direction = []
polygonImage = np.zeros((picHeight, picWidth, 3), np.uint8)
# Camera parameters
focal_length = 300.0
center_of_projection_x = 767
center_of_projection_y = 850

def main():
    initializeVariablesFromFiles()
    createVideoSequence()

def createVideoSequence():
    file_name = "result_"
    file_extension = ".jpg"
    frame_count = 0
    for i in range(20):
        r_vector = (0,0,0)
        t_vector = (0,5,-i/2.0)
        resultImg = processImage(r_vector, t_vector, frame_count)
        cv2.imwrite(file_name + str(frame_count) + file_extension, resultImg)
        frame_count += 1
        initializeVariablesFromFiles()

# Initialize arrays given 2 input file obtained using InputInterface.py
# points.txt will be sliced into 2 arrays: one containing 2d points, and the other containing the corresponding 3d points
# planeDetails.txt will be sliced into 2 arrays: one containing number of points in one plane, and the other containing the plane direction
def initializeVariablesFromFiles():
    # Parsing points.txt to initialize 2d and 3d points array
    print "Parsing points.txt ..."
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
    print "Parsing planeDetails.txt ..."
    planeDetails = open('planeDetails.txt')
    temp_array = planeDetails.read().splitlines()
    for i in range(len(temp_array)):
        temp_line = temp_array[i].split()
        if len(temp_line) == 0:
            continue
        else:
            array_points_per_plane.append(int(temp_line[0]))
            array_plane_direction.append(str(temp_line[1]))
    planeDetails.close()

# For all planes, perform homography
def processImage(r_vector, t_vector, sequence_number):
    print "Processing sequence" + str(sequence_number)
    resultImg = np.zeros((picHeight ,picWidth))
    num_planes = len(array_points_per_plane)
    # For all planes
    for i in range(num_planes):
        # Obtain plane detail for i-th plane
        numPoints = array_points_per_plane[i]
        planeDirection = array_plane_direction[i]
        resultImg = performHomography(numPoints, planeDirection, resultImg, r_vector, t_vector)
    return resultImg

# Given plane details, function will generate all possible 3d points and attach pixel color found through warped transformation
def performHomography(numPoints, planeDirection, resultImg, r_vector, t_vector):
    print "Performing homography for plane"
    array_points_2d = array_2d_points_raw[:numPoints]
    array_points_3d = array_3d_points_raw[:numPoints]
    del array_2d_points_raw[:numPoints]
    del array_3d_points_raw[:numPoints]

    if len(array_points_2d) == 0 or len(array_points_3d) == 0:
        return resultImg

    xStart = array_points_2d[0][0]
    xEnd = array_points_2d[1][0]
    yStart = array_points_2d[0][1]
    yEnd = array_points_2d[3][1]
    tempImg = cropImage(baseImage, xStart, xEnd, yStart, yEnd)

    array_points = getCorners(tempImg)
    camera = np.matrix([[focal_length, 0, center_of_projection_x],[0, focal_length, center_of_projection_y],[0, 0, 1]])
    resultPoints = getProjectedPoints(np.array(array_points_3d, np.float32), r_vector, t_vector, camera, 0)
    warppedImage = warpHomo(np.array(array_points, np.float32), np.array(resultPoints, np.float32), (picWidth, picHeight ), tempImg)

    duplicateImage = resultImg.copy()
    resultImg = overlayImage(duplicateImage, warppedImage)
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
    return cv2.warpPerspective(croppedImg, H[0], outputImgSize)

def overlayImage(resultImage, homoImage):
    greyHomo = cv2.cvtColor(homoImage, cv2.cv.CV_BGR2GRAY)
    retVal, greyHomo2 = cv2.threshold(greyHomo, 0,255,cv2.cv.CV_THRESH_BINARY)
    greyHomo_inv = ~greyHomo2

    newResultImage = np.zeros((picHeight,  picWidth, 3), np.uint8)
    newResultImage[:] = (0, 0, 0)
    newHomoImage = np.zeros(( picHeight, picWidth , 3), np.uint8)
    newHomoImage[:] = (0, 0, 0)

    cv2.bitwise_and(resultImage, resultImage, newResultImage,greyHomo_inv)
    cv2.bitwise_and(homoImage, homoImage, newHomoImage,greyHomo2)

    resultImage = newResultImage + newHomoImage
    return resultImage

main()