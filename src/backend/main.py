import openpyxl
import os
from JPGProcessingModule.main import main as jpg_main
from PDFProcessingModule.main import main as pdf_main


def main(source_path, output_directory_path, output_file_name):
    workbook = openpyxl.Workbook()
    workbook.remove(workbook.worksheets[0])

    if source_path[-4:] == ".pdf":
        pdf_main(source_path, workbook)
    elif source_path[-4:] in (".png", ".jpg") or source_path[-5:] == ".jpeg":
        jpg_main(source_path, workbook)
    else:
        raise ValueError("Wrong file type")

    workbook.save(os.path.join(output_directory_path, output_file_name))


if __name__ == "__main__":
    src = "PDFProcessingModule/pdf_tests/5_9_merged.pdf"
    out = "output"
    file_name = "text.xlsx"
    main(src, out, file_name)