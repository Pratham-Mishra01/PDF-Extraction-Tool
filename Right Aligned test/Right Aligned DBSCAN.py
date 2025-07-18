import cv2
import numpy as np
import pytesseract
import pandas as pd
from pdf2image import convert_from_path
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def preprocess_pdf_to_image(pdf_path, poppler_path):
    # Convert first page of PDF to image
    pages = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
    original_img = np.array(pages[0])  # RGB image

    # Convert to grayscale
    gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)

    # Threshold with inversion (black bg, white text) for dilation
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return original_img, gray, binary

def visualize_image(title, img, cmap='gray'):
    # Visualization using matplotlib
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.show()

def detect_cell_boxes(binary, median_width=None):
    # Calculate character widths using contours on the binary image
    char_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    widths = [cv2.boundingRect(c)[2] for c in char_contours if cv2.boundingRect(c)[2] > 2]

    median_width = int(np.median(widths)) if widths else (median_width or 15)  # Fallback to 15 if empty (useful when image is blur or empty)

    # Horizontal Kernel based on median width
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (median_width, 1))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # Find grouped word/cell blobs after dilation
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # width of word blobs
    widths2 = [cv2.boundingRect(c)[2] for c in contours if cv2.boundingRect(c)[2] > 2]  # collecting widths of words in an array
    median_width2 = int(np.median(widths2)) if widths2 else 15

    heights = [cv2.boundingRect(c)[3] for c in contours if cv2.boundingRect(c)[3] > 2]  # array of heights
    median_height = int(np.median(heights)) if heights else 15

    # avoiding page headers
    cell_boxes = [cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[3] < 1.5 * median_height]
    return cell_boxes, median_width2, median_height

def pad_cell_boxes(cell_boxes, min_height=20, image_height=None):
    # Force minimum height after box merging because tesseract wasn't detecting '-'
    padded = []
    for x, y, w, h in cell_boxes:
        pad_top = max(0, (min_height - h) // 2)
        pad_bottom = max(0, min_height - h - pad_top)
        y_new = max(0, y - pad_top)
        h_new = h + pad_top + pad_bottom
        if image_height:
            y_new = min(y_new, image_height - 1)
            h_new = min(h_new, image_height - y_new)
        padded.append((x, y_new, w, h_new))
    return padded

def draw_boxes_on_image(image, boxes):
    # Visualize on white background image
    vis_image = image.copy()
    for x, y, w, h in boxes:  # in order to see the padded boxes detected
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return vis_image

def run_ocr_on_cells(image, boxes):
    ocr_cells = []
    for i, (x, y, w, h) in enumerate(boxes):
        # using padded boxes dimensions to crop from og img
        pad = 2  # this padding was used because boxes were too close to the content and OCR detected 2 as 7
        y1 = max(0, y - pad)
        y2 = min(image.shape[0], y + h + pad)  # ensures that padding doesn't exceed lower image boundary
        cell_crop = image[y1:y2, x:x + w]

        # Preprocess for Tesseract
        cell_gray = cv2.cvtColor(cell_crop, cv2.COLOR_BGR2GRAY)  # RGB to grayscale
        cell_thresh = cv2.threshold(cell_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  # high contrast img for tesseract

        # CHOOSE PSM based on cell size
        config = '--oem 1 --psm 10' if w <= 20 and h <= 20 else '--oem 1 --psm 6'

        try:
            text = pytesseract.image_to_string(cell_thresh, config=config).strip()
        except Exception as e:
            print(f"OCR failed at cell {i} (x={x}, y={y}, w={w}, h={h}): {e}")
            text = ""

        # Save result
        ocr_cells.append({'x': x, 'y': y, 'w': w, 'h': h, 'text': text})
    return ocr_cells

def organize_cells_into_table(ocr_cells):
    # --- Row Binning ---
    row_tolerance = 15
    rows_dict = {}

    for cell in ocr_cells:
        y = cell['y']
        h = cell['h']
        mid = y + h // 2
        inserted = False

        if rows_dict:
            closest = min(rows_dict.keys(), key=lambda r: abs(mid - r))
            if abs(closest - mid) < row_tolerance:
                rows_dict[closest].append(cell)
                cell['row_anchor'] = closest
                inserted = True

        if not inserted:
            rows_dict[mid] = [cell]
            cell['row_anchor'] = mid

    # --- Column Anchors using (x + w) ---
    anchors = [cell['x'] + cell['w'] for cell in ocr_cells]
    anchors_sorted = sorted(anchors)
    col_gaps = np.diff(anchors_sorted)

    # Adaptive eps (safe range)
    raw_eps = np.percentile(col_gaps, 85) if len(col_gaps) > 0 else 10
    adaptive_eps = min(max(raw_eps, 5), 25)
    X = np.array(anchors).reshape(-1, 1)
    db = DBSCAN(eps=adaptive_eps, min_samples=3).fit(X)

    # Column centers from DBSCAN
    col_centers = [int(np.mean([anchors[i] for i in range(len(anchors)) if db.labels_[i] == label]))
                   for label in set(db.labels_) if label != -1]
    if not col_centers:
        col_centers = sorted(set(anchors))  # fallback
    col_centers.sort()

    # Assign col_anchor
    for cell in ocr_cells:
        center = cell['x'] + cell['w']
        best = min(col_centers, key=lambda c: abs(center - c))
        cell['col_anchor'] = best

    # --- Build Column Dictionary ---
    col_dict = {}
    for cell in ocr_cells:
        col_anchor = cell['col_anchor']
        if col_anchor not in col_dict:
            col_dict[col_anchor] = []
        col_dict[col_anchor].append(cell)

    row_mids = sorted(rows_dict.keys())
    col_mids = sorted(col_dict.keys())

    # mapping mids to their indices
    row_map = {item: idx for idx, item in enumerate(row_mids)}  # we are creating a dict of the type {mid: index}
    col_map = {item: idx for idx, item in enumerate(col_mids)}  # in python dictionaries are used in place of map data structure

    # Correct table initialization
    table = [['' for _ in col_mids] for _ in row_mids]

    # Fill the table using closest row/col anchors that were already stored in the cell
    for cell in ocr_cells:
        row_ind = row_map[cell['row_anchor']]
        col_ind = col_map[cell['col_anchor']]
        table[row_ind][col_ind] = cell['text']

    return table

def process_pdf_to_excel(pdf_path, poppler_path, output_excel='structured_table.xlsx'):
    original_img, gray, binary = preprocess_pdf_to_image(pdf_path, poppler_path)

    # Visualization 1: Grayscale
    visualize_image("Grayscale Image", gray)

    # Visualization 2: Binary (black bg, white text)
    visualize_image("Binary Image (Black BG, White Text)", binary)

    cell_boxes, median_width, median_height = detect_cell_boxes(binary)
    padded_boxes = pad_cell_boxes(cell_boxes, min_height=20, image_height=original_img.shape[0])

    vis_img = draw_boxes_on_image(original_img, padded_boxes)
    visualize_image("Step 3: Cell-Level Boxes via Adaptive Horizontal Dilation", cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB), cmap=None)

    ocr_cells = run_ocr_on_cells(original_img, padded_boxes)
    table = organize_cells_into_table(ocr_cells)

    # step 6
    df = pd.DataFrame(table)
    df.to_excel(output_excel, index=False, header=False)
    print("saved to the file structured_table")
