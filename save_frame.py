import cv2

cap = cv2.VideoCapture("test_small.mp4")
ret, frame = cap.read()
cv2.imwrite("frame.jpg", frame)
cap.release()

print("Saved as frame.jpg")
