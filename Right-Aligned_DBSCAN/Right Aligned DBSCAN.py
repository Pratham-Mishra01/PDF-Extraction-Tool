import cv2
import numpy as np
import pytesseract
import pandas as pd
from pdf2image import convert_from_path
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Set up paths
pdf_path = r"C:\Users\prath\OneDrive\Desktop\Pandas\OCR_DEMO\Kris_Love_Ltd_-_Gross_To_Net.pdf"
poppler_path = r"C:\Users\prath\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin"

def preprocess_pdf_to_images(pdf_path, poppler_path):
    # Convert all pages of PDF to images
    pages = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
    images = [np.array(p) for p in pages]  # List of RGB images
    return images

def preprocess_single_image(original_img):
    # Convert to grayscale
    gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    # Threshold with inversion (black bg, white text) for dilation
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return gray, binary

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
    median_width = int(np.median(widths)) if widths else (median_width or 15)

    
    # Horizontal dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (median_width, 1))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # Find grouped word/cell blobs after dilation
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    widths2 = [cv2.boundingRect(c)[2] for c in contours if cv2.boundingRect(c)[2] > 2]  # collecting widths of words in an array
    median_width2 = int(np.median(widths2)) if widths2 else 15
    heights = [cv2.boundingRect(c)[3] for c in contours if cv2.boundingRect(c)[3] > 2]
    median_height = int(np.median(heights)) if heights else 15

    cell_boxes = [cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[3] < 1.5 * median_height]

    # merging the multiline headers
    x_thresh = 1.5 * median_width2
    y_thresh = 1.5 * median_height
    used = [False] * len(cell_boxes)

    # merged=[]
    # for i in range(len(cell_boxes)):
    #     if used[i]:
    #         continue
    #     x1,y1,w1,h1=cell_boxes[i]
    #     if w1>(median_width2*10):
    #         continue
    #     box_group=[cell_boxes[i]]
    #     used[i]=True
    #     for j in range(i+1,len(cell_boxes)):
    #         if used[j]:
    #             continue
    #         x2,y2,w2,h2=cell_boxes[j]
    #         if abs(h1-h2)>median_height:
    #             continue
    #         if w2>(median_width2*10):
    #             continue
    #         if abs(x2-x1)<x_thresh and abs(y2-y1)<y_thresh:
    #             box_group.append(cell_boxes[j])
    #             used[j]=True
    #     x_vals=[b[0] for b in box_group]
    #     y_vals=[b[1] for b in box_group]
    #     x_end_vals=[b[0]+b[2] for b in box_group]
    #     y_end_vals=[b[1]+b[3] for b in box_group]
    #     x=min(x_vals)
    #     y=min(y_vals)
    #     w=max(x_end_vals)-x
    #     h=max(y_end_vals)-y
    #     merged.append((x,y,w,h))
    # cell_boxes=merged

    cell_boxes = sorted(cell_boxes, key=lambda b: (b[1], b[0]))
    return cell_boxes, median_width2, median_height


def pad_cell_boxes(cell_boxes, min_height=20, image_height=None):
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
    vis_image = image.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return vis_image

def run_ocr_on_cells(image, boxes):
    ocr_cells = []
    for i, (x, y, w, h) in enumerate(boxes):
        pad = 4
        y1 = max(0, y - pad)
        y2 = min(image.shape[0], y + h + pad)
        cell_crop = image[y1:y2, x:x + w]
        cell_gray = cv2.cvtColor(cell_crop, cv2.COLOR_BGR2GRAY)
        cell_thresh = cv2.threshold(cell_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        config ='--oem 1 --psm 6'
        try:
            text = pytesseract.image_to_string(cell_thresh, config=config).strip()
        except Exception as e:
            print(f"OCR failed at cell {i} (x={x}, y={y}, w={w}, h={h}): {e}")
            text = ""
        ocr_cells.append({'x': x, 'y': y, 'w': w, 'h': h, 'text': text})
    return ocr_cells

def cluster_rows_and_columns(ocr_cells):
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
    anchors = [cell['x'] + cell['w'] for cell in ocr_cells]
    anchors_sorted = sorted(anchors)
    col_gaps = np.diff(anchors_sorted)
    raw_eps = np.percentile(col_gaps, 85) if len(col_gaps) > 0 else 10
    adaptive_eps = min(max(raw_eps, 5), 25)
    X = np.array(anchors).reshape(-1, 1)
    db = DBSCAN(eps=adaptive_eps, min_samples=3).fit(X)
    col_centers = [int(np.mean([anchors[i] for i in range(len(anchors)) if db.labels_[i] == label]))
                   for label in set(db.labels_) if label != -1]
    col_centers.sort()
    for cell in ocr_cells:
        center = cell['x'] + cell['w']
        best = min(col_centers, key=lambda c: abs(center - c))
        cell['col_anchor'] = best
    return ocr_cells, rows_dict

def build_table(ocr_cells, rows_dict):
    col_dict = {}
    for cell in ocr_cells:
        col_anchor = cell['col_anchor']
        if col_anchor not in col_dict:
            col_dict[col_anchor] = []
        col_dict[col_anchor].append(cell)
    row_mids = sorted(rows_dict.keys())
    col_mids = sorted(col_dict.keys())
    row_map = {item: idx for idx, item in enumerate(row_mids)}
    col_map = {item: idx for idx, item in enumerate(col_mids)}
    table = [['' for _ in col_mids] for _ in row_mids]
    for cell in ocr_cells:
        row_ind = row_map[cell['row_anchor']]
        col_ind = col_map[cell['col_anchor']]
        table[row_ind][col_ind] = cell['text']
    return table

def save_all_tables_to_excel(tables, filename="Kris_love_ltd_Right Aligned DBSSCAN.xlsx"):
    writer = pd.ExcelWriter(filename, engine='openpyxl')
    for page_num, table in enumerate(tables, start=1):
        df = pd.DataFrame(table)
        df.to_excel(writer, sheet_name=f"Page_{page_num}", index=False, header=False)
    writer.close()
    print(f"Saved to {filename}")

if __name__ == "__main__":
    images = preprocess_pdf_to_images(pdf_path, poppler_path)
    all_tables = []
    
    for page_num, image in enumerate(images, start=1):
        gray, binary = preprocess_single_image(image)
        visualize_image(f"Page {page_num}: Grayscale", gray)
        visualize_image(f"Page {page_num}: Binary (Black BG, White Text)", binary)
        cell_boxes, median_width2, median_height = detect_cell_boxes(binary)
        padded_boxes = pad_cell_boxes(cell_boxes, min_height=20, image_height=image.shape[0])
        vis_image = draw_boxes_on_image(image, padded_boxes)
        visualize_image(f"Page {page_num}: Detected Cell Boxes", cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB), cmap=None)
        ocr_cells = run_ocr_on_cells(image, padded_boxes)
        ocr_cells, rows_dict = cluster_rows_and_columns(ocr_cells)
        table = build_table(ocr_cells, rows_dict)
        all_tables.append(table)
    
    save_all_tables_to_excel(all_tables, filename="Kris_love_ltd_Right Aligned DBSSCAN.xlsx")
