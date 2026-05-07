import pdfplumber


class ImageExtractor:

    def __init__(self, pdf_page: pdfplumber.pdf.Page):
        self._pdf_page = pdf_page

    def extract_images(self):
        raw_images = self._pdf_page.images
        images = []

        for raw_image in raw_images:
            bounding_box = (raw_image['x0'], raw_image['top'], raw_image['x1'], raw_image['bottom'])

            image = self._pdf_page.within_bbox(bounding_box).to_image(resolution=200)
            images.append(image)

        return images
