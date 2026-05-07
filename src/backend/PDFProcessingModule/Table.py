import pdfplumber.table


class Table:

    def __init__(self, raw_table: pdfplumber.table.Table, name=""):
        self.name = name

        self._raw_table = raw_table
        self._raw_cells = raw_table.cells
        self._content = raw_table.extract()
        self._intersection_cells = None
        self._positions = None

    def process_table(self):
        self._calculate_intersection_cells()
        self._calculate_positions()

    def _calculate_intersection_cells(self):
        vertical_lines = set()
        horizontal_lines = set()

        for cell in self._raw_cells:
            vertical_lines.add(cell[0])
            vertical_lines.add(cell[2])
            horizontal_lines.add(cell[1])
            horizontal_lines.add(cell[3])

        intersection_points = []
        for horizontal_line in horizontal_lines:
            for vertical_line in vertical_lines:
                intersection_points.append((vertical_line, horizontal_line))

        intersection_points = sorted(intersection_points, key=lambda x: x[::-1])
        self._intersection_cells = []
        for i in range(len(horizontal_lines) - 1):
            row = []
            for j in range(len(vertical_lines) - 1):
                row.append((intersection_points[i * len(vertical_lines) + j][0],
                            intersection_points[i * len(vertical_lines) + j][1],
                            intersection_points[(i + 1) * len(vertical_lines) + j + 1][0],
                            intersection_points[(i + 1) * len(vertical_lines) + j + 1][1]))
            self._intersection_cells.append(row)

    def _calculate_positions(self):
        self._positions = []
        for i in range(len(self._intersection_cells)):
            self._positions.append([])
            for j in range(len(self._intersection_cells[0])):
                self._positions[-1].append([i, j])

                parent_cell = None
                for raw_cell in self._raw_cells:
                    if raw_cell[0] <= self._intersection_cells[i][j][0] and \
                            raw_cell[1] <= self._intersection_cells[i][j][1] and \
                            raw_cell[2] >= self._intersection_cells[i][j][2] and \
                            raw_cell[3] >= self._intersection_cells[i][j][3]:

                        parent_cell = raw_cell
                        break

                if parent_cell is None:
                    continue

                flag = False
                for check_i in range(len(self._intersection_cells)):
                    if flag:
                        break
                    for check_j in range(len(self._intersection_cells[0])):
                        if self._content[check_i][check_j] is None:
                            continue
                        if self._intersection_cells[check_i][check_j][0] == parent_cell[0] and \
                                self._intersection_cells[check_i][check_j][1] == parent_cell[1]:

                            self._positions[check_i][check_j][0] = max(self._positions[check_i][check_j][0],
                                                                       i)
                            self._positions[check_i][check_j][1] = max(self._positions[check_i][check_j][1],
                                                                       j)
                            flag = True
                            break

    def save_table(self, worksheet):
        if self._intersection_cells is None or self._positions is None:
            raise ValueError("Process the table first")

        for i in range(len(self._content)):
            for j in range(len(self._content[0])):
                start_x = j + 1
                start_y = i + 1
                end_y = self._positions[i][j][0] + 1
                end_x = self._positions[i][j][1] + 1
                cell_content = self._content[i][j]
                if cell_content is None:
                    continue

                if end_x > start_x or end_y > start_y:
                    worksheet.merge_cells(
                        start_row=start_y,
                        start_column=start_x,
                        end_row=end_y,
                        end_column=end_x
                    )
                worksheet.cell(row=start_y, column=start_x).value = cell_content
