# Rule-Based Table Extraction from Scanned PDFs
This project extracts tables from scanned PDFs without gridlines using a pure rule-based pipeline built on OpenCV and Tesseract OCR. It is designed to handle documents with clean text layouts and no visual table lines.
# Warning:
  1. Does not work for Kris Love pdf because of vertical dilation is present in this code and multiline header code has been commented out
  2. Removing vertical dilation yields better results for Kris Love pdf, but the model does not print any word with "i" in the excel file as it detects the dot as a separate entity.
  3. The results given by this code contain many more errors from tesseract because the padding was set to 2. For Right Aligned and Row based I increased the padding to 4 and the errors from Tesseract were reduced significantly.

 ## Pipeline:
  1. PDF to Image Conversion
Each page is converted into a high-resolution image using pdf2image.

2. Image Preprocessing
Convert to grayscale.

Invert and binarize using Otsu thresholding.

3. Adaptive Dilation
Compute median character width using small contours.

Apply horizontal dilation to merge characters in the same row.

Apply vertical dilation to connect components like the dot on i.

4. Cell Detection
Find bounding boxes from contours after dilation.

Filter and pad them to ensure visibility for OCR.

5. OCR
Each cell box is processed with Tesseract (--psm 6) to extract text.

6. Row Grouping
Rows are grouped using the vertical midpoint of each cell.

Cells with similar midpoints (within a tolerance) are considered part of the same row.

7. Column Clustering (Right-Aligned)
In this pipeline, col_dict contains a tuple of (x,x+w) as the key.
Columns are grouped based on both:

x (left edge)

x + w (right edge)

If either edge is close to an existing column anchor, the cell is assigned to that column.

This helps avoid over-segmentation (e.g., separating i from a word).

8. Table Construction
A 2D array is created using the row and column mappings.

Text is inserted into the correct row/column location.

9. Excel Output
All extracted tables are written into a multi-sheet Excel file.

Each sheet corresponds to a page in the original PDF.


## Color Coding For Excel Results:
1. Red is for errors made by Tesseract.
2. Yellow is for cells that have been displaced.
3. Blue is for columns that have been displaced from the column header.

