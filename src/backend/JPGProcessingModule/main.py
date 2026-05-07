from JPGProcessingModule.JPGPreprocessor import JPGPreprocessor
from JPGProcessingModule.TableExtractor import TableExtractor
import cv2
import pytesseract


def main(source_path, workbook):
    pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract\installed\tesseract.exe'

    img = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
    preprocessor = JPGPreprocessor(img)  
    done_image = preprocessor.get_preprocessed_image()

    extractor = TableExtractor(done_image)
    tables = extractor.get_tables()

    for table in tables:
        table.process_table(done_image)

        sheet = workbook.create_sheet(table.name)
        table.save_table(sheet)

