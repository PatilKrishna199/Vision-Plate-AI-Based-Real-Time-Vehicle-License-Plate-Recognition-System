import cv2
import numpy as np
from skimage.filters import threshold_local
from skimage import measure
import imutils
import os

def sort_cont(character_contours):
    """
    Sort contours left to right
    """
    boundingBoxes = [cv2.boundingRect(c) for c in character_contours]
    (character_contours, boundingBoxes) = zip(
        *sorted(zip(character_contours, boundingBoxes),
                key=lambda b: b[1][0], reverse=False)
    )
    return character_contours


def segment_chars(plate_img, fixed_width):
    """
    Segment characters from detected license plate
    (Optimized for Tesseract OCR)
    """

    # Take Value channel (best for thresholding)
    V = cv2.split(cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV))[2]

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        V, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    thresh = cv2.bitwise_not(thresh)

    # Resize to canonical width
    plate_img = imutils.resize(plate_img, width=fixed_width)
    thresh = imutils.resize(thresh, width=fixed_width)

    # Connected component analysis
    labels = measure.label(thresh, background=0)
    charCandidates = np.zeros(thresh.shape, dtype='uint8')

    characters = []

    for label in np.unique(labels):
        if label == 0:
            continue

        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255

        cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1] if imutils.is_cv3() else cnts[0]

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)

            # Character validation rules
            aspectRatio = boxW / float(boxH)
            solidity = cv2.contourArea(c) / float(boxW * boxH)
            heightRatio = boxH / float(plate_img.shape[0])

            keepAspect = aspectRatio < 1.0
            keepSolidity = solidity > 0.15
            keepHeight = 0.5 < heightRatio < 0.95

            if keepAspect and keepSolidity and keepHeight and boxW > 14:
                hull = cv2.convexHull(c)
                cv2.drawContours(charCandidates, [hull], -1, 255, -1)

    # Extract all characters
    contours, _ = cv2.findContours(charCandidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contours = sort_cont(contours)
        addPixel = 4

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)

            x = max(x - addPixel, 0)
            y = max(y - addPixel, 0)

            temp = thresh[y:y + h + (addPixel * 2),
                          x:x + w + (addPixel * 2)]

            # return grayscale (best for Tesseract)
            characters.append(temp)

        return characters

    return None


class PlateFinder:
    def __init__(self, minPlateArea, maxPlateArea, resize=False):
        self.min_area = minPlateArea
        self.max_area = maxPlateArea
        self.element_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (22, 3))

    def preprocess(self, input_img):
        imgBlurred = cv2.GaussianBlur(input_img, (7, 7), 0)
        gray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)

        _, threshold_img = cv2.threshold(sobelx, 0, 255,
                                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        morph = threshold_img.copy()
        cv2.morphologyEx(threshold_img, cv2.MORPH_CLOSE,
                         self.element_structure, dst=morph)

        return morph

    def extract_contours(self, after_preprocess):
        contours, _ = cv2.findContours(after_preprocess,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        return contours

    def clean_plate(self, plate):
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY,
                                       11, 2)

        contours, _ = cv2.findContours(thresh.copy(),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)

        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            max_idx = np.argmax(areas)
            max_cnt = contours[max_idx]
            max_area = areas[max_idx]

            x, y, w, h = cv2.boundingRect(max_cnt)

            if not self.ratioCheck(max_area, plate.shape[1], plate.shape[0]):
                return plate, False, None

            return plate, True, [x, y, w, h]

        return plate, False, None

    def check_plate(self, input_img, contour):
        min_rect = cv2.minAreaRect(contour)

        if self.validateRatio(min_rect):
            x, y, w, h = cv2.boundingRect(contour)
            plate_region = input_img[y:y + h, x:x + w]

            clean_plate, found, coords = self.clean_plate(plate_region)

            if found:
                characters = self.find_characters_on_plate(clean_plate)

                if characters is not None and len(characters) >= 3:  # more flexible for Tesseract
                    cx, cy, cw, ch = coords
                    final_coords = (cx + x, cy + y)
                    return clean_plate, characters, final_coords

        return None, None, None

    def find_possible_plates(self, input_img):
        plates = []
        self.char_on_plate = []
        self.corresponding_area = []

        processed = self.preprocess(input_img)
        possible_cnts = self.extract_contours(processed)

        for c in possible_cnts:
            plate, chars, coords = self.check_plate(input_img, c)

            if plate is not None:
                plates.append(plate)
                self.char_on_plate.append(chars)
                self.corresponding_area.append(coords)

        return plates if plates else None

    def find_characters_on_plate(self, plate):
        return segment_chars(plate, 400)

    # RATIO CHECKS
    def ratioCheck(self, area, width, height):
        ratio = width / float(height)
        if ratio < 1:
            ratio = 1 / ratio

        return (self.min_area < area < self.max_area) and (3 < ratio < 6)

    def preRatioCheck(self, area, width, height):
        ratio = width / float(height)
        if ratio < 1:
            ratio = 1 / ratio

        return (self.min_area < area < self.max_area) and (2.5 < ratio < 7)

    def validateRatio(self, rect):
        (x, y), (width, height), angle = rect

        if width > height:
            ang = -angle
        else:
            ang = 90 + angle

        if ang > 15:
            return False
        if width == 0 or height == 0:
            return False

        area = width * height

        return self.preRatioCheck(area, width, height)
