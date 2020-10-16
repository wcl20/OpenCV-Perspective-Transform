import cv2
import imutils
import numpy as np
from warping import transform

points = []

def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONUP:
        points.append([x, y])

def main():

    global points

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_callback)

    image = cv2.imread("image.jpg")

    while True:

        # Crop image when all 4 corners are defined
        if len(points) == 4:
            image = transform(image, np.array(points))
            points = []

        # Draw points
        for point in points:
            image = cv2.circle(image, (point[0], point[1]), 3, (0, 255, 0), 1)

        cv2.imshow("Image", image)
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
