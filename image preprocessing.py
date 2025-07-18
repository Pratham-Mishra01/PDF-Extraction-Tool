import cv2
import numpy as np
import matplotlib.pyplot as plt
from pdf2image import convert_from_path
import pytesseract
from sklearn.cluster import DBSCAN

# Set up paths
pdf_path = r"C:\Users\prath\.vscode\Python Programs\sbP-cornerways\sb&P-cornerways\Cornerways Payroll 2024-2025.pdf"
poppler_path = r"C:\Users\prath\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin"

def convert_pdf_to_image(pdf_path, poppler_path):
    # Convert first page of PDF to image
    pages = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
    return np.array(pages[0])  # RGB image

def preprocess_image(original_img):
    # Convert to grayscale
    gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)

    # Threshold with inversion (black bg, white text) for dilation
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return gray, binary

def compute_median_width(binary):
    # Calculate character widths using contours on the binary image
    char_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    widths = [cv2.boundingRect(c)[2] for c in char_contours if cv2.boundingRect(c)[2] > 2]
    return int(np.median(widths)) if widths else 15  # Fallback to 15 if empty (useful when image is blur or empty)

def detect_cell_boxes(binary, median_width):
    # Horizontal Kernel based on median width
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (median_width, 1))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # Find grouped word/cell blobs after dilation
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # width of word blobs
    widths2 = [cv2.boundingRect(c)[2] for c in contours if cv2.boundingRect(c)[2] > 2]  # collecting widths of words in an array
    heights = [cv2.boundingRect(c)[3] for c in contours if cv2.boundingRect(c)[3] > 2]  # array of heights
    median_height = int(np.median(heights)) if heights else 15

    cell_boxes = [cv2.boundingRect(c) for c in contours 
                  if cv2.boundingRect(c)[3] < 1.5 * median_height]  # avoiding page headers

    # merging the multiline headers
    # x_thresh=1.5*median_width2  # threshold for merging
    # y_thresh=1.5*median_height
    # used=[False]*len(cell_boxes)  # arr of False value equal to the number of boxes in cell_boxes
    # merged=[]
    # for i in range(len(cell_boxes)):
    #     if used[i]:
    #         continue
    #
    #     x1,y1,w1,h1=cell_boxes[i]
    #     if w1>(median_width2*10):
    #         continue
    #     box_group=[cell_boxes[i]]
    #     used[i]=True
    #     for j in range(i+1,len(cell_boxes)):
    #         if used[j]:
    #             continue
    #         x2,y2,w2,h2=cell_boxes[j]
    #         if abs(h1-h2)>median_height:  # merging boxes of similar height
    #             continue
    #         if w2>(median_width2*10):
    #             continue
    #         if abs(x2-x1)<x_thresh and abs(y2-y1)<y_thresh:
    #             box_group.append(cell_boxes[j])
    #             used[j]=True
    #     # new dimensions
    #     x_vals=[b[0] for b in box_group]  # list with x values of all the boxes in box_grp
    #     y_vals=[b[1] for b in box_group]
    #     x_end_vals=[b[0]+b[2] for b in box_group]
    #     y_end_vals=[b[1]+b[3] for b in box_group]
    #     x=min(x_vals)
    #     y=min(y_vals)
    #     w=max(x_end_vals)-x
    #     h=max(y_end_vals)-y
    #     merged.append((x,y,w,h))
    # cell_boxes=merged

    cell_boxes = sorted(cell_boxes, key=lambda b: (b[1], b[0]))  # in tables we should first sort top to bottom to obtain rows and then left to right
    return cell_boxes, median_height

def pad_boxes(cell_boxes, min_height=20):
    # Force minimum height after box merging because tesseract wasnt detecting '-'
    padded_boxes = []       
    for x, y, w, h in cell_boxes:
        pad_top = max(0, (min_height - h) // 2)
        pad_bottom = max(0, min_height - h - pad_top)
        y_new = max(0, y - pad_top)
        h_new = h + pad_top + pad_bottom
        padded_boxes.append((x, y_new, w, h_new))
    return padded_boxes

def visualize_boxes(img, boxes, title):
    # Visualize on white background image
    vis_img = img.copy()
    for x, y, w, h in boxes:  # in order to see the padded boxes detected
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    plt.figure(figsize=(15, 15))
    plt.title(title)
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def perform_ocr(padded_boxes, original_img):
    ocr_cells = []
    for i, (x, y, w, h) in enumerate(padded_boxes):
        # using padded boxes dimensions to crop from og img
        pad = 2  # this padding was used because boxes were too close to the content and OCR detected 2 as 7
        y1 = max(0, y - pad)
        y2 = min(original_img.shape[0], y + h + pad)  # ensures that padding doesnt exceed lower image boundary
        cell_crop = original_img[y1:y2, x:x + w]

        # Preprocess for Tesseract
        cell_gray = cv2.cvtColor(cell_crop, cv2.COLOR_BGR2GRAY)  # RGB to grayscale
        cell_thresh = cv2.threshold(cell_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  # high contrast img for tesseract

        # CHOOSE PSM based on cell size
        if w <= 20 and h <= 20:
            config = '--oem 1 --psm 10'
        else:
            config = '--oem 1 --psm 6'

        try:
            text = pytesseract.image_to_string(cell_thresh, config=config).strip()
        except Exception as e:
            print(f"OCR failed at cell {i} (x={x}, y={y}, w={w}, h={h}): {e}")
            text = ""

        # Save result
        ocr_cells.append({'x': x, 'y': y, 'w': w, 'h': h, 'text': text})
    return ocr_cells

def run_pipeline():
    original_img = convert_pdf_to_image(pdf_path, poppler_path)
    gray, binary = preprocess_image(original_img)

    # Visualization 1: Grayscale
    plt.figure(figsize=(10, 6))
    plt.title("Grayscale Image")
    plt.imshow(gray, cmap='gray')
    plt.axis('off')
    plt.show()

    # Visualization 2: Binary (black bg, white text)
    plt.figure(figsize=(10, 6))
    plt.title("Binary Image (Black BG, White Text)")
    plt.imshow(binary, cmap='gray')
    plt.axis('off')
    plt.show()

    median_width = compute_median_width(binary)
    cell_boxes, _ = detect_cell_boxes(binary, median_width)
    padded_boxes = pad_boxes(cell_boxes)

    visualize_boxes(original_img, padded_boxes, "Step 3: Cell-Level Boxes via Adaptive Horizontal Dilation")

    ocr_cells = perform_ocr(padded_boxes, original_img)
    return ocr_cells

if __name__ == '__main__':
    ocr_cells = run_pipeline()
