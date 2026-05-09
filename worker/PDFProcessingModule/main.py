import os

import pdfplumber
from PDFProcessingModule.TableExtractor import TableExtractor
from PDFProcessingModule.ImageExtractor import ImageExtractor
from JPGProcessingModule.main import main as jpg_main
import uuid


def main(source_path, workbook):
    pdf = pdfplumber.open(source_path)
    for page in pdf.pages:
        table_extractor = TableExtractor(page)
        image_extractor = ImageExtractor(page)

        images = image_extractor.extract_images()
        tables = table_extractor.extract_tables()

        for table in tables:
            table.process_table()

            sheet = workbook.create_sheet("From_PDF")
            table.save_table(sheet)

        for image in images:
            image_path = os.path.join(os.environ["TEMP_FOLDER"], str(uuid.uuid4()) + ".jpg")
            image.save(image_path)
            jpg_main(image_path, workbook)
            os.remove(image_path)
