import numpy as np
import cv2
from shapely.geometry import LineString
from JPGProcessingModule.ContentRecognizer import ContentRecognizer


class Table:
    class Config:
        DELTA = 6

    def __init__(self, rotated_cells):
        self.cells_count = len(rotated_cells)

        self._rotated_cells = rotated_cells
        self._boxed_cells = np.array([cv2.boxPoints(rotated_cell) for rotated_cell in self._rotated_cells],
                                     dtype=np.int64)
        self._content = None
        self._positions = None

    def process_table(self, preprocessed_image):
        self._calculate_cells_position()
        self._calculate_cells_content(preprocessed_image)

    def save_table(self, worksheet):
        if self._positions is None or self._content is None:
            raise ValueError("Process the table first")

        for i in range(self.cells_count):
            start_x, end_x, start_y, end_y = self._positions[i]
            cell_content = self._content[i]

            if end_x > start_x or end_y > start_y:
                worksheet.merge_cells(
                    start_row=start_y,
                    start_column=start_x,
                    end_row=end_y,
                    end_column=end_x
                )
            worksheet.cell(row=start_y, column=start_x).value = cell_content

    def _calculate_cells_position(self):
        self._positions = [0] * self.cells_count

        horizontal_distances = []
        vertical_distances = []

        bound = cv2.boxPoints(cv2.minAreaRect(self._boxed_cells.reshape(-1, 2))).astype(int)
        bound_corners = Table._get_rect_corners(bound)

        upper_bound = LineString((bound_corners[0], bound_corners[2]))
        left_bound = LineString((bound_corners[0], bound_corners[1]))

        horizontal_distance_by_cell_ind = [0] * self.cells_count
        vertical_distance_by_cell_ind = [0] * self.cells_count

        horizontal_len_by_cell_ind = [0] * self.cells_count
        vertical_len_by_cell_ind = [0] * self.cells_count

        for i in range(self.cells_count):
            cell = self._boxed_cells[i]

            cell_corners = Table._get_rect_corners(cell)

            vertical_line = LineString(
                (bound_corners[0] - bound_corners[1] + cell_corners[0], cell_corners[0]))

            upper_intersection_point = np.array(upper_bound.intersection(vertical_line).coords, dtype=np.int64)

            horizontal_distance = round(((bound_corners[0] - upper_intersection_point) ** 2).sum() ** 0.5)
            horizontal_len = round(((cell_corners[2] - cell_corners[0]) ** 2).sum() ** 0.5)

            for j in range(len(horizontal_distances)):
                if -Table.Config.DELTA < horizontal_distance - horizontal_distances[j] < Table.Config.DELTA:
                    horizontal_distance_by_cell_ind[i] = horizontal_distances[j]
                    break
            else:
                horizontal_distances.append(horizontal_distance)
                horizontal_distance_by_cell_ind[i] = horizontal_distance
            for j in range(len(horizontal_distances)):
                if -Table.Config.DELTA < horizontal_distance_by_cell_ind[i] + horizontal_len - horizontal_distances[j] \
                        < Table.Config.DELTA:
                    horizontal_len_by_cell_ind[i] = horizontal_distances[j] - horizontal_distance_by_cell_ind[i]
                    break
            else:
                horizontal_distances.append(horizontal_distance + horizontal_len)
                horizontal_len_by_cell_ind[i] = horizontal_len

            horizontal_line = LineString(
                (bound_corners[0] - bound_corners[2] + cell_corners[0], cell_corners[0]))

            left_intersection_point = np.array(left_bound.intersection(horizontal_line).coords, dtype=np.int64)

            vertical_distance = round(((bound_corners[0] - left_intersection_point) ** 2).sum() ** 0.5)
            vertical_len = round(((cell_corners[1] - cell_corners[0]) ** 2).sum() ** 0.5)

            for j in range(len(vertical_distances)):
                if -Table.Config.DELTA < vertical_distance - vertical_distances[j] < Table.Config.DELTA:
                    vertical_distance_by_cell_ind[i] = vertical_distances[j]
                    break
            else:
                vertical_distances.append(vertical_distance)
                vertical_distance_by_cell_ind[i] = vertical_distance
            for j in range(len(vertical_distances)):
                if -Table.Config.DELTA < vertical_distance_by_cell_ind[i] + vertical_len - vertical_distances[j] < \
                        Table.Config.DELTA:
                    vertical_len_by_cell_ind[i] = vertical_distances[j] - vertical_distance_by_cell_ind[i]
                    break
            else:
                vertical_distances.append(vertical_distance + vertical_len)
                vertical_len_by_cell_ind[i] = vertical_len

        horizontal_distances = sorted(horizontal_distances)
        vertical_distances = sorted(vertical_distances)

        for i in range(self.cells_count):
            horizontal_distance = horizontal_distance_by_cell_ind[i]
            vertical_distance = vertical_distance_by_cell_ind[i]

            horizontal_len = horizontal_len_by_cell_ind[i]
            vertical_len = vertical_len_by_cell_ind[i]

            start_x, end_x, start_y, end_y = None, None, None, None
            for j in range(len(horizontal_distances)):
                if -Table.Config.DELTA < horizontal_distance - horizontal_distances[j] < Table.Config.DELTA \
                        and start_x is None:
                    start_x = j + 1
                if -Table.Config.DELTA < horizontal_distance + horizontal_len - horizontal_distances[j] < \
                        Table.Config.DELTA \
                        and end_x is None:
                    end_x = j
            for j in range(len(vertical_distances)):
                if -Table.Config.DELTA < vertical_distance - vertical_distances[j] < Table.Config.DELTA \
                        and start_y is None:
                    start_y = j + 1
                if -Table.Config.DELTA < vertical_distance + vertical_len - vertical_distances[j] < Table.Config.DELTA \
                        and end_y is None:
                    end_y = j
            self._positions[i] = start_x, end_x, start_y, end_y

    def _calculate_cells_content(self, preprocessed_image):
        self._content = ContentRecognizer.calculate_content(self._rotated_cells, preprocessed_image)

    @staticmethod
    def _get_rect_corners(rect):
        sorted_rect = rect[np.lexsort((rect[:, 0], rect[:, 1]))]
        if sorted_rect[0][0] < sorted_rect[1][0]:
            left_upper_corner = sorted_rect[0]
            left_down_corner = sorted_rect[2]
            right_upper_corner = sorted_rect[1]
            right_down_corner = sorted_rect[3]
        else:
            left_upper_corner = sorted_rect[1]
            left_down_corner = sorted_rect[3]
            right_upper_corner = sorted_rect[0]
            right_down_corner = sorted_rect[2]
        return left_upper_corner, left_down_corner, right_upper_corner, right_down_corner

    # DEBUG
    def _show_table(self, image):
        blank = np.uint8(np.zeros(image.shape))

        c = [np.int64(cv2.boxPoints(r)) for r in self._rotated_cells]
        bound = cv2.boxPoints(cv2.minAreaRect(self._boxed_cells.reshape(-1, 2))).astype(int)
        c.append(bound)

        cv2.drawContours(blank, c, -1, (255, 255, 255), 1)

        cv2.imshow("", blank)
        cv2.waitKey(0)

    # DEBUG
    @staticmethod
    def _show_cell(cell):
        cv2.imshow("", cell)
        cv2.waitKey(0)
