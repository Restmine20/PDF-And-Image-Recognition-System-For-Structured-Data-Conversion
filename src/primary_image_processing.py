import cv2
import numpy as np
import math


# PDF -> JPEG converter (not implemented)
def pdf_to_image_conversion():
    pass


# Primary JPEG processing (centralization, scaling, perspective change, contrast)
def primary_image_processing(image):
    is_vertical_orientation, document_contour = sort_contour_coordinates(
        approx_contour(
            find_max_contour(
                find_document_contours(image))))

    width, height = None, None
    if is_vertical_orientation:
        width = 600
        height = int(math.sqrt(2) * width)
    else:
        height = 600
        width = int(math.sqrt(2) * height)

    output_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    matrix = cv2.getPerspectiveTransform(document_contour, output_points)

    return cv2.warpPerspective(img, matrix, (width, height))


# Find document closed contour (rectangle)
def find_document_contours(image):
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
    _, thresh_image = cv2.threshold(blurred_image, 130, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


# Extract the biggest contour from list
def find_max_contour(contours):
    max_contour = None
    for contour in contours:
        if max_contour is None or contour.shape[0] > max_contour.shape[0]:
            max_contour = contour
    return max_contour


# Approximating a contour to a rectangle
def approx_contour(contour):
    arc_length = cv2.arcLength(contour, True)

    eps = 0.05
    epsilon = arc_length * eps

    approximate_contour = cv2.approxPolyDP(contour, epsilon, True)
    return approximate_contour.squeeze()


# Find a document orientation and choose corners
def sort_contour_coordinates(contour):
    answer_contour = np.float32([[-1, -1], [-1, -1], [-1, -1], [-1, -1]])

    sorted_contour_by_y = contour[np.lexsort((contour[:, 0], contour[:, 1]))]
    is_vertical_orientation = None

    if sorted_contour_by_y[0][0] == sorted_contour_by_y[3][0]:
        is_vertical_orientation = True
    else:
        tan = abs((sorted_contour_by_y[3][1] - sorted_contour_by_y[0][1]) /
                  (sorted_contour_by_y[3][0] - sorted_contour_by_y[0][0]))
        is_vertical_orientation = tan >= 1

    if sorted_contour_by_y[0][0] < sorted_contour_by_y[1][0]:
        answer_contour[0] = sorted_contour_by_y[0]
        answer_contour[1] = sorted_contour_by_y[1]
    else:
        answer_contour[0] = sorted_contour_by_y[1]
        answer_contour[1] = sorted_contour_by_y[0]
    if sorted_contour_by_y[3][0] < sorted_contour_by_y[2][0]:
        answer_contour[2] = sorted_contour_by_y[3]
        answer_contour[3] = sorted_contour_by_y[2]
    else:
        answer_contour[2] = sorted_contour_by_y[2]
        answer_contour[3] = sorted_contour_by_y[3]

    return is_vertical_orientation, answer_contour