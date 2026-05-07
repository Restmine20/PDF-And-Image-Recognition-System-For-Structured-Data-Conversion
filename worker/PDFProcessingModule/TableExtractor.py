import pdfplumber
from PDFProcessingModule.Table import Table


class TableExtractor:
    class Config:
        SETTINGS = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 3,
        }

    def __init__(self, pdf_page: pdfplumber.pdf.Page):
        self._pdf_page = pdf_page

    def extract_tables(self):
        raw_tables = self._pdf_page.find_tables(TableExtractor.Config.SETTINGS)
        return [Table(raw_table) for raw_table in raw_tables]
