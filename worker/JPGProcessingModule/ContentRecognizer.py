from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import os
import cv2
import tesserocr
import threading


class ContentRecognizer:
    thread_data = threading.local()

    class Config:
        X_AXIS_CELL_SCALE = 2
        Y_AXIS_CELL_SCALE = 2

    @staticmethod
    def calculate_content(rotated_cells, preprocessed_image):
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            return list(executor.map(lambda x: ContentRecognizer._get_cell_content(preprocessed_image, x),
                                     rotated_cells))

    @staticmethod
    def _preprocess_cell(preprocessed_image, rotated_cell):
        bound_rect = cv2.boundingRect(cv2.boxPoints(rotated_cell))
        cell = preprocessed_image[bound_rect[1]: bound_rect[1] + bound_rect[3],
                                  bound_rect[0]: bound_rect[0] + bound_rect[2]]

        if rotated_cell[-1] != -90:
            rot_matrix = cv2.getRotationMatrix2D((bound_rect[2] // 2, bound_rect[3] // 2),
                                                 rotated_cell[-1], 1.0)
            cell = cv2.warpAffine(cell, rot_matrix, (bound_rect[2], bound_rect[3]))

        cell = cv2.resize(cell, None, fx=ContentRecognizer.Config.X_AXIS_CELL_SCALE,
                          fy=ContentRecognizer.Config.Y_AXIS_CELL_SCALE,
                          interpolation=cv2.INTER_LANCZOS4)

        # ТУТ ДИНАМИЧЕСКИ МЕНЯТЬ РАМКУ, ЧТОБЫ КОНТУР НЕ ПОПАДАЛ, ЧИСЛО ЧИСТО ПОДГОН ДЛЯ ПРИМЕРА
        cell = cell[2: cell.shape[0] - 2, 2: cell.shape[1] - 2]

        return Image.fromarray(cell)

    @staticmethod
    def _get_cell_content(preprocessed_image, rotated_cell):
        cell_image = ContentRecognizer._preprocess_cell(preprocessed_image, rotated_cell)

        ocr_api = ContentRecognizer._get_ocr_api()

        ocr_api.SetImage(cell_image)

        return ocr_api.GetUTF8Text()

    @staticmethod
    def _get_ocr_api():
        if not hasattr(ContentRecognizer.thread_data, "ocr_api"):
            ContentRecognizer.thread_data.ocr_api = tesserocr.PyTessBaseAPI(lang="rus+eng",
                                                                            psm=tesserocr.PSM.SINGLE_BLOCK,
                                                                            oem=tesserocr.OEM.LSTM_ONLY)
        return ContentRecognizer.thread_data.ocr_api
