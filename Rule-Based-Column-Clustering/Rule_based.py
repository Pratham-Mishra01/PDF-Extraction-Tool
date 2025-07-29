import cv2
import numpy as np
import pytesseract
import pandas as pd
from pdf2image import convert_from_path
import matplotlib.pyplot as plt

# -------------------------------
# CONFIGURATION
# -------------------------------
PDF_PATH = r"C:\Users\prath\OneDrive\Desktop\Pandas\OCR_DEMO\Kris_Love_Ltd_-_Gross_To_Net.pdf"
POPLER_PATH = r"C:\Users\prath\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin"
DPI = 300

# -------------------------------
# FUNCTIONS
# -------------------------------
def convert_pdf_to_images(pdf_path, poppler_path, dpi=300):
    pages = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
    return [np.array(p) for p in pages]

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return gray, binary

def adaptive_dilation(binary):
    char_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    widths = [cv2.boundingRect(c)[2] for c in char_contours if cv2.boundingRect(c)[2] > 2]
    median_width = int(np.median(widths)) if widths else 15

    # Horizontal dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (median_width, 1))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # Vertical dilation
    kernel_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    dilated = cv2.dilate(dilated, kernel_vert, iterations=1)

    return dilated

def extract_cell_boxes(dilated):
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    widths = [cv2.boundingRect(c)[2] for c in contours if cv2.boundingRect(c)[2] > 2]
    heights = [cv2.boundingRect(c)[3] for c in contours if cv2.boundingRect(c)[3] > 2]

    median_width = int(np.median(widths)) if widths else 15
    median_height = int(np.median(heights)) if heights else 15

    cell_boxes = [cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[3] < 1.5 * median_height]
    cell_boxes = sorted(cell_boxes, key=lambda b: (b[1], b[0]))

    # Pad for OCR visibility
    MIN_HEIGHT = 20
    padded_boxes = []
    for x, y, w, h in cell_boxes:
        pad_top = max(0, (MIN_HEIGHT - h) // 2)
        pad_bottom = max(0, MIN_HEIGHT - h - pad_top)
        y_new = max(0, y - pad_top)
        h_new = h + pad_top + pad_bottom
        padded_boxes.append((x, y_new, w, h_new))

    return padded_boxes

def perform_ocr(boxes, original_img):
    ocr_cells = []
    for i, (x, y, w, h) in enumerate(boxes):
        pad = 2
        y1 = max(0, y - pad)
        y2 = min(original_img.shape[0], y + h + pad)
        crop = original_img[y1:y2, x:x + w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        config = '--oem 1 --psm 6'
        try:
            text = pytesseract.image_to_string(thresh, config=config).strip()
        except:
            text = ""

        ocr_cells.append({'x': x, 'y': y, 'w': w, 'h': h, 'text': text})
    return ocr_cells

def cluster_rows(ocr_cells, tolerance=15):
    rows_dict = {}
    for cell in ocr_cells:
        y, h = cell['y'], cell['h']
        mid = y + h // 2
        inserted = False
        if rows_dict:
            closest = min(rows_dict.keys(), key=lambda r: abs(mid - r))
            if abs(closest - mid) < tolerance:
                rows_dict[closest].append(cell)
                cell['row_anchor'] = closest
                inserted = True
        if not inserted:
            rows_dict[mid] = [cell]
            cell['row_anchor'] = mid
    return rows_dict

def cluster_columns(ocr_cells, tolerance=10):
    col_dict = {}
    for cell in ocr_cells:
        x, w = cell['x'], cell['w']
        right = x + w
        best_key = None
        min_error = float('inf')

        # check if the cell matches any existing column anchor
        for key in col_dict.keys():
            left_key, right_key = key
            left_error = abs(x - left_key)
            right_error = abs(right - right_key)

            if left_error <= tolerance or right_error <= tolerance:
                total_error = min(left_error, right_error)
                if total_error < min_error:
                    min_error = total_error
                    best_key = key

        # if a close match was found, assign it to that column
        if best_key:
            col_dict[best_key].append(cell)
            cell['col_anchor'] = best_key
        else:
            # else create new key with the left and right x coords
            key_tuple = (x, right)
            col_dict[key_tuple] = [cell]
            cell['col_anchor'] = key_tuple

    return col_dict


def construct_table(ocr_cells, rows_dict, col_dict):
    row_mids = sorted(rows_dict.keys())
    col_mids = sorted(col_dict.keys(), key=lambda k: k[0])  # Always sort by x, the left edge
    row_map = {item: idx for idx, item in enumerate(row_mids)}
    col_map = {item: idx for idx, item in enumerate(col_mids)}

    table = [['' for _ in col_mids] for _ in row_mids]
    for cell in ocr_cells:
        row = row_map[cell['row_anchor']]
        col = col_map[cell['col_anchor']]
        table[row][col] = cell['text']
    return table

def export_all_tables(tables, filename="Kris love Ltd_ Rule based.xlsx"):
    writer = pd.ExcelWriter(filename, engine='openpyxl')
    for i, table in enumerate(tables):
        df = pd.DataFrame(table)
        df.to_excel(writer, sheet_name=f"Page_{i+1}", index=False, header=False)
    writer.close()
    print(f"Saved to {filename}")

# -------------------------------
# MAIN EXECUTION
# -------------------------------
if __name__ == "__main__":
    images = convert_pdf_to_images(PDF_PATH, POPLER_PATH, dpi=DPI)
    all_tables = []

    for page_num, img in enumerate(images):
        print(f"Processing page {page_num + 1}...")
        gray, binary = preprocess_image(img)
        dilated = adaptive_dilation(binary)
        boxes = extract_cell_boxes(dilated)
        ocr_cells = perform_ocr(boxes, img)
        rows_dict = cluster_rows(ocr_cells)
        col_dict = cluster_columns(ocr_cells)
        table = construct_table(ocr_cells, rows_dict, col_dict)
        all_tables.append(table)

    export_all_tables(all_tables)
