import cv2

cap = cv2.VideoCapture("test_small.mp4")

if not cap.isOpened():
    print("FAIL: Cannot open video")
    exit()

print("SUCCESS: Video opened")

ret, frame = cap.read()
print("First frame read:", ret)

cap.release()
