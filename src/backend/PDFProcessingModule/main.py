import pdfplumber
from PDFProcessingModule.TableExtractor import TableExtractor
from PDFProcessingModule.ImageExtractor import ImageExtractor


def main(source_path, workbook):
    pdf = pdfplumber.open(source_path)

    table_num = 0
    for page in pdf.pages:
        table_extractor = TableExtractor(page)
        image_extractor = ImageExtractor(page)

        images = image_extractor.extract_images()
        tables = table_extractor.extract_tables()

        for table in tables:
            table.name = f"Table {table_num}"
            table_num += 1

            table.process_table()

            sheet = workbook.create_sheet(table.name)
            table.save_table(sheet)

        for image in images:
            # отправить на обработку в JPGModule
            pass
