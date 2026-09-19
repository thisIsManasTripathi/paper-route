import pymupdf
import numpy as np
import cv2
from paddleocr import PaddleOCR

doc = pymupdf.open("targets/scanned.pdf")

page = doc[0]

pix = page.get_pixmap()

img = np.frombuffer(pix.samples, dtype=np.uint8)
img = img.reshape(pix.height, pix.width, pix.n)

# Paddle/OpenCV generally works with BGR
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

ocr = PaddleOCR(enable_mkldnn=False, lang='en', use_angle_cls=True)

result = ocr.predict(img) #iterable: a list, llength : 1! (just the single Result object)

#we are concerned with 
# print(result[0])
# print(type(result[0]))
print(len(result))