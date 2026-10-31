import io
import pymupdf

from PIL import Image

from app.services.ocr_service import extract_text_from_image


def extract_pdf_pages(pdf_path):

    extracted_pages = []

    document = pymupdf.open(pdf_path)

    try:
        for page_number, page in enumerate(document, start=1):

            text = page.get_text("text").strip()

            extraction_method = "text"

            # Use OCR for scanned or nearly empty pages.
            if len(text) < 30:

                pixmap = page.get_pixmap(
                    matrix=pymupdf.Matrix(2, 2),
                    alpha=False
                )

                image_bytes = pixmap.tobytes("png")

                image = Image.open(
                    io.BytesIO(image_bytes)
                )

                text = extract_text_from_image(image)

                extraction_method = "ocr"

            if text.strip():

                extracted_pages.append({
                    "page_number": page_number,
                    "text": text,
                    "extraction_method": extraction_method
                })

    finally:
        document.close()

    return extracted_pages