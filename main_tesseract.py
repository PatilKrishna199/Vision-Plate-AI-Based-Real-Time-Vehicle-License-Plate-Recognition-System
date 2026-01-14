import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Create virtual environment:-


def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 150)

    # Close gaps (very important for number plates)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    return closed


def extract_plate(img):
    edged = preprocess(img)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort largest contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h)

        # Indian car plate approx ratio (2.5–6)
        if 2.2 < aspect < 6.5 and w > 80 and h > 20:
            plate = img[y:y+h, x:x+w]
            return plate, (x, y, w, h)

    return None, None


def ocr_plate(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 31, 15
    )

    config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    try:
        return pytesseract.image_to_string(thresh, config=config).strip()
    except:
        return ""


# -------- Main Program ---------
cap = cv2.VideoCapture("test_small.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    plate_img, box = extract_plate(frame)

    if plate_img is None or plate_img.size == 0:
        print("Skipped invalid plate ROI")
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # VALID PLATE FOUND → run OCR
    text = ocr_plate(plate_img)

    if len(text) > 3:
        print("Detected Plate:", text)

    # Draw bounding box
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow("Plate", plate_img)
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
