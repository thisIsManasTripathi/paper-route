import os
import fitz
import cv2
import numpy as np

PDF_DIR = "targets"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(PDF_DIR):
    if not filename.lower().endswith(".pdf"):
        continue

    pdf_path = os.path.join(PDF_DIR, filename)

    doc = fitz.open(pdf_path)
    page = doc[0]

    pix = page.get_pixmap()
    img = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img.reshape(pix.height, pix.width, pix.n)



    # Crop top 30% of the page
    h, w = img.shape[:2]
    crop = img[:int(h * 0.30), :]

    output_name = os.path.splitext(filename)[0] + "_crop.png"
    output_path = os.path.join(OUTPUT_DIR, output_name)

    cv2.imwrite(output_path, crop)

    doc.close()

    print(f"{filename} -> {output_path}")