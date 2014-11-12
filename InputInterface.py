import cv2
import cv2.cv as cv

'''
Program will create a user interface that allows users to click on points on the image.
Users can then specify a 3D coordinate corresponding to that point clicked.
The program will store the clicked point, the 3d point as well as the plane information into 2 files.
points.txt contains a list of array with 5 elements, first 2 being the image points clicked, last 3 being the 3d corresponding points.
planeDetails.txt contains a list of 2 elements, first one specifying the number of points in a plane, second being the direction of the plane.
'''

pointsFile = open('points.txt','a')
planeFile = open('planeDetails.txt', 'a')

def onMouse(event, x, y, flags, param):

    if event == cv.CV_EVENT_LBUTTONDOWN:
        print "Clicked at point: [" + str(x) + ", " + str(y) + "]"
        estimated_3D_point = raw_input("Please enter estimated 3D point (e.g. 100 100 100): ")
        array_3d_points = estimated_3D_point.split()

        pointsFile.write(str(x) + " ")
        pointsFile.write(str(y) + " ")
        pointsFile.write(array_3d_points[0] + " ")
        pointsFile.write(array_3d_points[1] + " ")
        pointsFile.write(array_3d_points[2])
        pointsFile.write("\n");

        userInput = raw_input("Done selecting corners for a plane? y/n ")
        if userInput == 'y':
            numPoints = raw_input("Please enter number of points chosen: ")
            plane_direction = raw_input("Please enter direction of normal for the plane (e.g. up, left, right, front, down): ")
            planeFile.write(numPoints + " " + plane_direction + "\n")

            userInput = raw_input("Done selecting all planes (y/n)? ")
            if userInput == 'y':
                cv2.destroyWindow("Picture")
                pointsFile.close()
                planeFile.close()
            else:
                return
        else:
            return

def printFileContent():
    data = list([line.strip() for line in open('points.txt')])
    print data
    data = list([line.strip() for line in open('planeDetails.txt')])
    print data

def main():
    image = cv2.imread("assets/project.jpeg", cv2.CV_LOAD_IMAGE_COLOR)

    cv2.namedWindow('Picture', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Picture', onMouse, 0)
    cv2.imshow("Picture", image)
    cv2.waitKey()

    printFileContent()

main()