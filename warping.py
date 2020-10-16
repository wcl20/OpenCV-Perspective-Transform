import cv2
import numpy as np

def order_points(points):
    '''
    Sort points to correct order (tl, tr, br, bl)
    '''
    # Create bounding box
    bbox = np.zeros((4, 2), dtype="float32")

    # Get Top-left and Bottom-right corners
    sum = points.sum(axis=1)
    bbox[0] = points[np.argmin(sum)]
    bbox[2] = points[np.argmax(sum)]

    # Get Top-right and Bottom-left corners
    diff = np.diff(points, axis=1)
    bbox[1] = points[np.argmin(diff)]
    bbox[3] = points[np.argmax(diff)]

    return bbox

def distance(pt1, pt2):
    d = np.sqrt(((pt1[0] - pt2[0]) ** 2) + ((pt1[1] - pt2[1]) ** 2))
    return int(d)

def transform(image, points):
    '''
    Apply warp transform
    '''
    # Get bounding box
    bbox = order_points(points)
    tl, tr, br, bl = bbox
    # Calculate dimensions
    width = max(distance(br, bl), distance(tr, tl))
    height = max(distance(tr, br), distance(tl, bl))
    # Apply transform
    dest = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(bbox, dest)
    return cv2.warpPerspective(image, M, (width, height))
