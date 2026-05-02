import cv2
import numpy as np
from shapely.geometry import Polygon
import collections
from JPGProcessingModule.Table import Table


class TableExtractor:
    class Config:
        MIN_CELL_RATIO = 1 / 1500
        MAX_CELL_COMPACT = 60
        GAUSSIAN_BLUR_KERNEL = 5
        THRESH_BLOCK_SIZE = 15
        THRESH_C = 7
        HORIZONTAL_KERNEL_RATIO = 1 / 40
        VERTICAL_KERNEL_RATIO = 1 / 40
        REPAIR_KERNEL_SIZE = 7
        DELTA = 5

    def __init__(self, preprocessed_image):
        self._preprocessed_image = preprocessed_image

    def get_tables(self):
        table_structure = self._find_tables_structure()
        rotated_cells = self._get_all_rotated_cells(table_structure)

        tables = TableExtractor._split_cells_to_tables(rotated_cells)

        return tables

    def _find_tables_structure(self):
        blurred_image = cv2.GaussianBlur(self._preprocessed_image,
                                         (TableExtractor.Config.GAUSSIAN_BLUR_KERNEL,
                                          TableExtractor.Config.GAUSSIAN_BLUR_KERNEL), 0)

        thresh_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV,
                                             TableExtractor.Config.THRESH_BLOCK_SIZE,
                                             TableExtractor.Config.THRESH_C)

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                                      (int(thresh_image.shape[1] *
                                                       TableExtractor.Config.HORIZONTAL_KERNEL_RATIO), 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                                    (1, int(thresh_image.shape[0] *
                                                     TableExtractor.Config.VERTICAL_KERNEL_RATIO)))
        # написать одной операцией OPENING
        horizontal_mask = cv2.erode(thresh_image, horizontal_kernel, iterations=1)
        horizontal_mask = cv2.dilate(horizontal_mask, horizontal_kernel, iterations=1)

        vertical_mask = cv2.erode(thresh_image, vertical_kernel, iterations=1)
        vertical_mask = cv2.dilate(vertical_mask, vertical_kernel, iterations=1)

        tables_structure = cv2.bitwise_or(horizontal_mask, vertical_mask)

        repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                                  (TableExtractor.Config.REPAIR_KERNEL_SIZE,
                                                   TableExtractor.Config.REPAIR_KERNEL_SIZE))

        return cv2.morphologyEx(tables_structure, cv2.MORPH_CLOSE, repair_kernel, iterations=1)

    def _get_all_rotated_cells(self, table_structure):
        contours, hierarchy = cv2.findContours(table_structure, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        filtered_indexes = []
        filtered_rotated_rects = []

        total_square = self._preprocessed_image.shape[0] * self._preprocessed_image.shape[1]

        for i in range(len(contours)):
            if cv2.contourArea(contours[i]) > TableExtractor.Config.MIN_CELL_RATIO * total_square and \
                    cv2.arcLength(contours[i], True) ** 2 / cv2.contourArea(contours[i]) < \
                    TableExtractor.Config.MAX_CELL_COMPACT:

                filtered_indexes.append(i)
                filtered_rotated_rects.append(cv2.minAreaRect(contours[i]))

        rotated_cells = []

        for i in range(len(filtered_rotated_rects)):
            is_cell = True
            ind = hierarchy[0][filtered_indexes[i]][2]
            while ind != -1:
                if ind in filtered_indexes:
                    is_cell = False
                    break
                ind = hierarchy[0][ind][0]
            if is_cell:
                rotated_cells.append(filtered_rotated_rects[i])

        return rotated_cells

    @staticmethod
    def _split_cells_to_tables(rotated_cells):
        cells = np.array([np.int64(cv2.boxPoints(rotated_cell)) for rotated_cell in rotated_cells])

        graph_cells = [0] * len(cells)
        for i in range(len(cells)):
            graph_cells[i] = [0] * len(cells)

        for i in range(len(cells)):
            for j in range(len(cells)):
                if i == j:
                    graph_cells[i][j] = 1
                    continue

                rect_cell_i = Polygon(cells[i])
                rect_cell_j = Polygon(cells[j])

                distance = rect_cell_i.distance(rect_cell_j)

                if distance < TableExtractor.Config.DELTA:
                    graph_cells[i][j] = 1
                    graph_cells[j][i] = 1

        used_cells = [False] * len(cells)
        tables = []

        for i in range(len(cells)):
            if used_cells[i]:
                continue

            cur_raw_table = []
            cell_ind_queue = collections.deque()
            cell_ind_queue.append(i)

            while len(cell_ind_queue) != 0:
                cur_cell_ind = cell_ind_queue.popleft()

                if used_cells[cur_cell_ind]:
                    continue
                used_cells[cur_cell_ind] = True

                cur_raw_table.append(rotated_cells[cur_cell_ind])

                for neighbour in range(len(cells)):
                    if graph_cells[cur_cell_ind][neighbour] == 1:
                        cell_ind_queue.append(neighbour)

            cur_table_number = len(tables)
            tables.append(Table(cur_raw_table, cur_table_number))

        return tables
