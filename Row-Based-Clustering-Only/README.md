# Row-Based Table Extraction from Scanned PDFs (No Columns)
This repository contains a rule-based OCR pipeline designed to extract rows only from scanned PDF documents, especially where column structure is inconsistent or irrelevant.

🚀 Overview
This script extracts horizontal row-wise data from scanned PDFs by:

Converting pages to images

Preprocessing them to isolate characters

Grouping detected text boxes into rows based on vertical alignment

Exporting the extracted data to a multi-sheet Excel file

## Warning:
  1. Does not work for Kris Love ltd because of vertical dilation and removal of merging for multiline headers.
  2. Removing vertical dilation makes it work for Kris Love but the model then fails to detect words with the character "i" in them.
  3. Multiline headers segment is commented out because it significantly decreases the accuracy in other pdfs.
  4. Since it is only row based clustering the columns are displaced.
  5. Tesseract makes lesser mistakes because I increased the padding from 2 to 4.

✅ Features
🔍 Text-focused extraction — no reliance on table lines or column rules

📄 Handles multi-page PDFs

🧱 Uses OpenCV and Tesseract for processing

📊 Outputs clean Excel sheets — one per page

🧠 How It Works
1. PDF to Image Conversion
Each PDF page is converted to a high-resolution RGB image using pdf2image.

2. Preprocessing
Converted to grayscale

Inverted and binarized via Otsu thresholding

Dilation is applied:

Horizontal dilation joins nearby characters into words

Vertical dilation preserves vertical continuity

3. Cell Detection
Contours are extracted from the dilated image to find bounding boxes for potential text cells.

4. OCR Extraction
Each bounding box is processed individually with Tesseract OCR using --psm 6 (assumes a uniform block of text).

5. Row Clustering
The vertical midpoint of each cell is used to group them into rows.

Cells that lie close to each other vertically are placed in the same row.

6. Row Assembly
Cells in each row are sorted left to right (x-coordinate)

Final data is structured row-wise (no column inference)

7. Excel Export
Each page is saved as a separate sheet in an Excel file.

📦 Requirements
Install required dependencies:

bash
Copy
Edit
pip install opencv-python pdf2image pytesseract pandas openpyxl matplotlib
You also need:

Tesseract OCR installed and added to system path

Poppler for PDF-to-image conversion

⚙️ Configuration
Set the PDF path and Poppler binary path at the bottom of the script:

python
Copy
Edit
pdf_path = r"path\to\your\document.pdf"
poppler_path = r"path\to\poppler\bin"
You can change the output file name in:

python
Copy
Edit
process_pdf(pdf_path, poppler_path, output_excel="your_output_file.xlsx")
