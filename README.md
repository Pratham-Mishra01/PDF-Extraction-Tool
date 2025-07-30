# 🧠 Robust Table Extraction from Scanned PDFs using OpenCV + Tesseract
This repository provides three specialized pipelines for extracting tabular data from scanned PDFs using OpenCV for preprocessing and Tesseract for OCR. These rule-based systems are designed to handle various real-world layouts — with or without visible table lines — and are built for high-volume processing (~20,000 PDFs/month).
## 🔍 Pipeline Summary
1. Right Aligned DBSCAN- For PDFs where amounts (like salary, prices) are right-aligned but column boundaries are unclear.
2. Rule-Based- When full tabular structure is needed; performs cell-level detection and reconstructs a 2D table
3. Row-Only- For unstructured rows where column alignment is unreliable or unnecessary.

## ⚙️ Common Architecture
1. All pipelines follow a similar core flow:

2. PDF to Image: via pdf2image

3. Image Preprocessing: grayscale → threshold → dilation

4. Bounding Box Detection: using contours in OpenCV

5. OCR: Tesseract with adaptive PSM (usually --psm 6)
6. Clustering/Grouping:
      DBSCAN/Rule Based column Grouping
      Vertical alignment for Row Grouping.
7. Export: Multi-sheet Excel file (one sheet per PDF page)

## 🧩 Dependencies
Ensure the following Python packages and external tools are installed:

### 📦 Python Packages
You can install these via pip:

pip install opencv-python
pip install pytesseract
pip install numpy
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install pdf2image
pip install openpyxl

