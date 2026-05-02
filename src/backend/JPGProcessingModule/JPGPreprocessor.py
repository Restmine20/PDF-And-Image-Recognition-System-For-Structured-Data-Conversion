import cv2
import numpy as np
import math


class JPGPreprocessor:
    class Config:
        CANNY_SIGMA = 0.33
        GAUSSIAN_BLUR_KERNEL = 5
        APPROX_EPSILON = 0.03
        DOCUMENT_CONTOURS_TRIES = 3
        TAN_THRESHOLD = 2

    def __init__(self, raw_image, base_size=800):
        self._raw_image = raw_image
        self._base_size = base_size

    def get_preprocessed_image(self):
        try:
            is_vertical_orientation, document_contour = self._find_document_contour()

            if is_vertical_orientation:
                width = self._base_size
                height = int(math.sqrt(2) * width)
            else:
                height = self._base_size
                width = int(math.sqrt(2) * height)

            output_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

            perspective_matrix = cv2.getPerspectiveTransform(document_contour, output_points)

            # DEBUG
            #return self._raw_image
            return cv2.warpPerspective(self._raw_image, perspective_matrix, (width, height))
        except ValueError:
            return self._raw_image

    def _find_document_contour(self):
        image_median = int(np.median(self._raw_image))

        lower_threshold = int(max(0, (1 - JPGPreprocessor.Config.CANNY_SIGMA) * image_median))
        upper_threshold = int(min(255, (1 + JPGPreprocessor.Config.CANNY_SIGMA) * image_median))

        blurred_image = cv2.GaussianBlur(self._raw_image, (JPGPreprocessor.Config.GAUSSIAN_BLUR_KERNEL,
                                                           JPGPreprocessor.Config.GAUSSIAN_BLUR_KERNEL), 0)

        edged_image = cv2.Canny(blurred_image, lower_threshold, upper_threshold)

        contours, _ = cv2.findContours(edged_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours,
                          key=cv2.contourArea,
                          reverse=True)[: JPGPreprocessor.Config.DOCUMENT_CONTOURS_TRIES]

        for contour in contours:
            epsilon = cv2.arcLength(contour, True) * JPGPreprocessor.Config.APPROX_EPSILON
            approximated_contour = cv2.approxPolyDP(contour, epsilon, True).squeeze()

            if approximated_contour.shape[0] == 4:
                return JPGPreprocessor._sort_contour_coordinates(approximated_contour)

        raise ValueError("Document not found")

    @staticmethod
    def _sort_contour_coordinates(contour):
        answer_contour = np.float32([[-1, -1], [-1, -1], [-1, -1], [-1, -1]])

        sorted_contour_by_y = contour[np.lexsort((contour[:, 0], contour[:, 1]))]

        if sorted_contour_by_y[0][0] == sorted_contour_by_y[3][0]:
            is_vertical_orientation = True
        else:
            tan = abs((sorted_contour_by_y[3][1] - sorted_contour_by_y[0][1]) /
                      (sorted_contour_by_y[3][0] - sorted_contour_by_y[0][0]))
            is_vertical_orientation = tan >= JPGPreprocessor.Config.TAN_THRESHOLD

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
