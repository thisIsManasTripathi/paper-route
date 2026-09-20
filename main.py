import pymupdf
import numpy as np
import cv2
from paddleocr import PaddleOCR
import re
from difflib import SequenceMatcher

courseCodes = [
    # First year (Sem I & II)
    "MAC01", "PHC01", "CYC01", "XEC01", "ESC01", "BTC01", "HSC01",
    "MAC02", "CSC01", "XEC02", "CSC02",

    # Biotechnology
    "MAC331", "CHC331", "BTC301", "BTC302", "BTC303",
    "BTC401", "CHC431", "BTC402", "CSC431", "HSC431",
    "BTC501", "BTC502", "BTC503", "CHC531",
    "XEC631", "CSC631", "CHC631", "MSC731",

    # Chemical Engineering
    "CHC301", "CHC302", "CHC303", "CYC331",
    "CHC401", "CHC402", "CHC403", "MEC431",
    "CHC501", "CHC502", "CHC503", "CHC504",
    "CHC601", "CHC602", "CHC603",

    # Civil Engineering
    "CEC301", "CEC302", "CEC303", "ESC331",
    "CEC401", "CEC402", "CEC403", "CSC432",
    "CEC501", "CEC502", "CEC503", "CEC504", "CEC505",
    "CEC601", "CEC602",

    # Computer Science and Engineering
    "CSC301", "CSC302", "CSC303", "PHC331",
    "CSC401", "CSC402", "CSC403", "CSC404", "CSC405",
    "CSC501", "CSC502", "CSC503", "CSC504",
    "CSC601", "CSC602",

    # Electrical Engineering
    "EEC301", "EEC302", "ECC331", "PHC332",
    "EEC401", "EEC402", "EEC403",
    "EEC501", "EEC502", "EEC503", "EEC504",
    "EEC601", "EEC602",

    # Electronics and Communication Engineering
    "ECC301", "ECC302", "ECC303",
    "ECC401", "ECC402", "ECC403", "EEC431",
    "ECC501", "ECC502", "ECC503", "ECC504",
    "ECC601", "ECC602",

    # Mechanical Engineering
    "MEC301", "MEC302", "MEC303", "MEC304", "PHC333",
    "MEC401", "MEC402", "MEC403", "EEC432",
    "MEC501", "MEC502", "MEC503", "MEC504",
    "MEC601", "MEC602", "MEC851",

    # Metallurgical and Materials Engineering
    "MMC301", "MMC302", "MMC303", "ESC332",
    "MMC401", "MMC402", "MMC403", "CSC433",
    "MMC501", "MMC502", "MMC503", "MMC504",
    "MMC601", "MMC602", "MMC851", "MMC852", "MMC853",

    # Information Technology
    "ITC301", "ITC302", "ITC303",
    "ITC401", "ITC402", "ITC403", "ITC404",
    "ITC501", "ITC502", "ITC503", "ITC504",
    "ITC601", "ITC602", "ITC851", "ITC852", "ITC853",

    # Chemistry (5-Year Integrated MSc)
    "CYC301", "CYC302", "CYC303", "PHC334",
    "CYC401", "CYC402", "CYC403", "CYC404",
    "CYC501", "CYC502", "CYC503", "CYC504",
    "CYC601", "CYC602", "CYC603",
    "CYC701", "CYC702", "CYC703", "CYC704",
    "CYC801", "CYC802", "CYC803",
]

def getCodeFallback(ambgCode: str):
    scores = [SequenceMatcher(None, ambgCode, code).ratio() for code in courseCodes]
    bestIndex = scores.index(max(scores))
    if scores[bestIndex] < 0.75:
        return None
    return courseCodes[bestIndex]


def cropImage(img):
    # Crop top 30% of the page
    h, w = img.shape[:2]
    crop = img[:int(h * 0.30), :]

    return crop

def getMetaData(img):
    ocr = PaddleOCR(enable_mkldnn=False, lang='en', use_textline_orientation=True)
    result = ocr.predict(img)[0] #iterable: a list, llength : 1! (just the single Result object)

    blob = " ".join(result['rec_texts'])

    metaData = dict.fromkeys(["year", "semester", "term", "code", "review"], None)
    metaData["review"] = False
    #we are concerned with rec_texts attribute
    # print(result, result.keys())
    print(blob)

    # check for year
    yearM = re.search(r'\b\d{4}-\d{2}\b', blob)
    if yearM:
        metaData['year'] = yearM.group().split("-")[0]
    else:
        metaData['review'] = True

    # check for code
    codeRex = "|".join(courseCodes)
    codeM = re.search(rf'{codeRex}', blob)
    if codeM:
        metaData['code'] = codeM.group()
    else:
        fbCodeRex = re.search(r'\b[A-Za-z]{3}\d{2,3}\b', blob)
        if fbCodeRex == None:
            metaData['review'] = True
        
        else:
            fbCode = getCodeFallback(fbCodeRex.group())
            if fbCode == None:
                metaData['review'] = True
            else:
                metaData['code'] = fbCode


    # check for term
    termM = re.search(r'(?i)\b(?:mid[-\s]?term|end[-\s]?term)\b', blob).group() # type: ignore
    if termM:
        metaData['term'] = termM[:3].lower()+"sem"
    else:
        metaData['review'] = True

    # getting sem from the subject code
    if metaData['code']:
        if (len(metaData['code'])) == 6:
            metaData['semester'] = int(metaData['code'][3])
        else:
             metaData['semester'] = int(metaData['code'][-1])
    else:
        metaData['review'] = True


    print(metaData)

    


def processImage(doc):
    page = doc[0]
    pix = page.get_pixmap()
    img = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img.reshape(pix.height, pix.width, pix.n)

    # Convert RGB/RGBA → BGR
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    croppedImage = cropImage(img)
    return croppedImage

        
doc = pymupdf.open("targets/image_warped1.pdf")
getMetaData(processImage(doc))



# print(type(result[0]))
# print(len(result))