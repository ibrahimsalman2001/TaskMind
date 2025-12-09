# ocr_module.py

import easyocr
from typing import List

# Initialize once
reader = easyocr.Reader(["en", "ur"], gpu=False)

def extract_text_from_frame(frame_path: str) -> str:
    """
    Uses EasyOCR to extract text from a single image frame.
    Returns joined text string.
    """
    result = reader.readtext(frame_path, detail=0)
    return " ".join(result)


def extract_text_from_frames(frame_paths: List[str]) -> str:
    """
    Loops through frames and extracts combined OCR text.
    """
    all_text = []
    for frame in frame_paths:
        text = extract_text_from_frame(frame)
        all_text.append(text)

    return " ".join(all_text)