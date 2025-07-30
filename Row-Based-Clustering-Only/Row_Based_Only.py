import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pytesseract
from pdf2image import convert_from_path
from openpyxl import Workbook
from openpyxl.writer.excel import save_workbook
import os

def convert_pdf_to_images(pdf_path, poppler_path, dpi=300):
    return convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return gray, binary

def adaptive_dilation(binary):
    char_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    widths = [cv2.boundingRect(c)[2] for c in char_contours if cv2.boundingRect(c)[2] > 2]
    median_width = int(np.median(widths)) if widths else 15

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (median_width, 1))
    dilated = cv2.dilate(binary, h_kernel, iterations=1)

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    dilated = cv2.dilate(dilated, v_kernel, iterations=1)

    return dilated

def find_cell_boxes(dilated):
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    widths2 = [cv2.boundingRect(c)[2] for c in contours if cv2.boundingRect(c)[2] > 2]
    heights = [cv2.boundingRect(c)[3] for c in contours if cv2.boundingRect(c)[3] > 2]
    median_width2 = int(np.median(widths2)) if widths2 else 15
    median_height = int(np.median(heights)) if heights else 15

    cell_boxes = [cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[3] < 1.5 * median_height]
    cell_boxes = sorted(cell_boxes, key=lambda b: (b[1], b[0]))

    MIN_HEIGHT = 20
    padded_boxes = []
    for x, y, w, h in cell_boxes:
        pad_top = max(0, (MIN_HEIGHT - h) // 2)
        pad_bottom = max(0, MIN_HEIGHT - h - pad_top)
        y_new = max(0, y - pad_top)
        h_new = h + pad_top + pad_bottom
        padded_boxes.append((x, y_new, w, h_new))
    return padded_boxes, median_width2, median_height

def extract_text_from_boxes(image, boxes):
    ocr_cells = []
    for i, (x, y, w, h) in enumerate(boxes):
        pad = 4
        y1 = max(0, y - pad)
        y2 = min(image.shape[0], y + h + pad)
        cell_crop = image[y1:y2, x:x + w]
        cell_gray = cv2.cvtColor(cell_crop, cv2.COLOR_BGR2GRAY)
        cell_thresh = cv2.threshold(cell_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        config = '--oem 1 --psm 6'
        try:
            text = pytesseract.image_to_string(cell_thresh, config=config).strip()
        except Exception as e:
            print(f"OCR failed at cell {i} (x={x}, y={y}, w={w}, h={h}): {e}")
            text = ""

        ocr_cells.append({'x': x, 'y': y, 'w': w, 'h': h, 'text': text})
    return ocr_cells

def group_cells_into_rows(ocr_cells, row_tolerance=15):
    rows_dict = {}
    for cell in ocr_cells:
        x = cell['x']
        w = cell['w']
        y = cell['y']
        h = cell['h']
        mid = y + h // 2
        inserted = False

        if rows_dict:
            closest_ry = min(rows_dict.keys(), key=lambda r: abs(mid - r))
            if abs(closest_ry - mid) < row_tolerance:
                rows_dict[closest_ry].append(cell)
                inserted = True

        if not inserted:
            rows_dict[mid] = [cell]

    sorted_rows = [rows_dict[key] for key in sorted(rows_dict.keys())]
    table_data = []
    for row in sorted_rows:
        sorted_cells = sorted(row, key=lambda c: c['x'])
        texts = [cell['text'] for cell in sorted_cells]
        table_data.append(texts)

    df = pd.DataFrame(table_data)
    return df

def process_pdf(pdf_path, poppler_path, output_excel="multi_page_tables.xlsx"):
    pages = convert_pdf_to_images(pdf_path, poppler_path)
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        for idx, page in enumerate(pages):
            original_img = np.array(page)
            gray, binary = preprocess_image(original_img)
            dilated = adaptive_dilation(binary)
            cell_boxes, _, _ = find_cell_boxes(dilated)
            ocr_cells = extract_text_from_boxes(original_img, cell_boxes)
            df = group_cells_into_rows(ocr_cells)
            df.to_excel(writer, sheet_name=f"Page_{idx + 1}", index=False, header=False)

    print(f"Saved row-based tables to '{output_excel}'")


# Example usage
pdf_path = r"C:\Users\prath\.vscode\Python Programs\sbP-cornerways\sb&P-cornerways\Cornerways Payroll 2024-2025.pdf"
poppler_path = r"C:\Users\prath\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin"
process_pdf(pdf_path, poppler_path)
