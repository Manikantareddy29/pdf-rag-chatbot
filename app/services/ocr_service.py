import pytesseract

from app.config import TESSERACT_CMD, OCR_LANGUAGE

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def extract_text_from_image(image):

    try:
        text = pytesseract.image_to_string(
            image,
            lang=OCR_LANGUAGE
        )

        return text.strip()

    except Exception as e:

        raise RuntimeError(
            f"OCR extraction failed. Check Tesseract installation: {e}"
        )