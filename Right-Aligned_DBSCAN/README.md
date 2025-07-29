# Table Extraction from Scanned PDFs with Right-Aligned DBSCAN for Column Clustering
This project implements a robust OCR pipeline to extract tables from scanned or image-based PDFs. It handles cases where:
    1.No visible gridlines are present
    2.Data is right-aligned (common in financial or payroll documents)

# WARNING TO DEVELOPERS:
1. Tesseract makes fewer mistakes in the results of right aligned when compared to Rule_Based primarily because I increased the top and bottom padding from 2 to 4
2. The code does not work for the Kris Love pdf primarily because of vertical dilation and because I have commented out the code for multiline header merging.
3. The absence of vertical dilation makes the model work for Kris Love pdf, but it affects words with "i". The model does not print words with the character "i" in them as it detects the dot as a separate entity.
4. Multiline header detection code has been commented out as although it works for Kris Love pdf, it significantly decreases the accuracy for the other pdfs.

## Key Features:
  1. Dynamic horizontal dilation based on median character width
  2. Right-aligned DBSCAN column clustering using x + w as anchor
  3. Row grouping via vertical anchors with tolerance
  4. Padded crops for better OCR accuracy
  5. Output to Excel (.xlsx)

## ⚙️ Pipeline Overview
  1. PDF to Image: Convert using pdf2image
  2. Preprocessing: Grayscale, binary thresholding
  3. Cell Detection:
      Horizontal dilation with median character width
      Vertical dilation to merge characters like "i" properly
  4. Bounding Box Extraction: Find contours on the dilated image
  5. OCR with Tesseract: Apply PSM 7 or 6 based on aspect ratio
  6. Column Clustering:
      Use right-edge (x + w) anchors
      Apply DBSCAN to identify column groups
  7. Row Grouping:
      Group by vertical positions with tolerance
      Assign each OCR'd cell to a (row, col) index
  8. Excel Export: Final reconstructed table exported as .xlsx

# Color Coding For Excel Results:
1. Used the color RED to highlight mistakes made by Tesseract Library in detecting texts. Most of the problems noticed were in detecting the decimal point.
2. Used the color YELLOW to highlight the cells that were misplaced by the model.
3. Used the color BLUE to highlight the columns that were shifted from the column header.
