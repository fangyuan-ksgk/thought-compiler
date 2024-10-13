from pdf2image import convert_from_path
import io
from PIL import Image
import fitz  # PyMuPDF
import numpy as np
from typing import List, Tuple
import json

def pdf_to_img(file_path, first_page=1, last_page=1):
    # Convert PDF pages to images
    img_pages = convert_from_path(file_path, first_page=first_page, last_page=last_page)

    # Calculate the total width and maximum height
    total_width = sum(page.width for page in img_pages)
    max_height = max(page.height for page in img_pages)

    # Create a new image with the calculated dimensions
    combined_img = Image.new('RGB', (total_width, max_height))

    # Paste each page into the combined image
    x_offset = 0
    for page in img_pages:
        combined_img.paste(page, (x_offset, 0))
        x_offset += page.width

    return combined_img

def file_to_img(file_path, first_page=1, last_page=1):
    if file_path.endswith(".pdf"):
        img = pdf_to_img(file_path, first_page, last_page)
    elif file_path.endswith(".eml"):
        img = None 
    elif file_path.endswith((".png", ".jpg", ".jpeg")):
        with Image.open(file_path) as img_raw:
            buffered = io.BytesIO()
            img_raw.save(buffered, format="PNG")
            img = buffered.getvalue()
    else:
        raise ValueError("Unknown file format")
    return img


def file_to_preprocessed_img(file_path, first_page=1, last_page=1):
    import base64

    if file_path.endswith(".pdf"):
        img = pdf_to_img(file_path, first_page, last_page)
    elif file_path.endswith(".eml"):
        return None
    elif file_path.endswith((".png", ".jpg", ".jpeg")):
        img = Image.open(file_path)
    else:
        raise ValueError("Unknown file format")

    # Convert image to PNG
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    
    # Convert PNG to base64
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return image_base64

def pdf_to_text_and_images(paper_path: str) -> Tuple[str, List[np.ndarray]]:
    """
    Convert PDF to text and images.
    """
    doc = fitz.open(paper_path)
    texts = []
    images = []
    
    for page in doc:
        texts.append(page.get_text())
        pix = page.get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        images.append(img)
    
    return texts, images[0]


def parse_citation_and_tags(response: str):
    try:
        import re 
        match = re.match(r'```json\n(.*?)\n```', response, re.DOTALL)
        result = json.loads(match.group(1))
        citations = result.get("citations", [])
        tags = result.get("tags", [])
        print(f"Extracted {len(citations)} citations and {len(tags)} tags.")
    except json.JSONDecodeError:
        print("Error: Unable to parse JSON response from OpenAI.")
        citations = []
        tags = []
    return citations, tags

def parse_citations(response: str):
    try:
        import re
        match = re.match(r'```json\n(.*?)\n```', response, re.DOTALL)
        if match:
            result = json.loads(match.group(1))
            citations = result.get("citations", [])
            print(f"Extracted {len(citations)} citations.")
            return citations
        else:
            print("Error: No JSON block found in the response.")
            return []
    except json.JSONDecodeError:
        print("Error: Unable to parse JSON response.")
        return []

def parse_tags(response: str):
    try:
        import re
        match = re.match(r'```json\n(.*?)\n```', response, re.DOTALL)
        if match:
            result = json.loads(match.group(1))
            tags = result.get("tags", [])
            print(f"Extracted {len(tags)} tags.")
            return tags
        else:
            print("Error: No JSON block found in the response.")
            return []
    except json.JSONDecodeError:
        print("Error: Unable to parse JSON response.")
        return []

import datetime

def get_time_range(begin_date="2018-01-01", end_date="now", interval="7d"):
    from datetime import datetime, timedelta
    
    end_date = datetime.now() if end_date == "now" else datetime.strptime(end_date, "%Y-%m-%d")
    begin_date = datetime.strptime(begin_date, "%Y-%m-%d")
    
    interval_days = int(interval[:-1])
    time_ranges = []
    
    current_start = begin_date
    while current_start < end_date:
        current_end = min(current_start + timedelta(days=interval_days), end_date)
        
        start_str = current_start.strftime("%Y-%m-%d%H%M%S")
        end_str = current_end.strftime("%Y-%m-%d%H%M%S")
        
        time_ranges.append((start_str, end_str))
        
        current_start = current_end
    
    return time_ranges

