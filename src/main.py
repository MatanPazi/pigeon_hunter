from picamera2 import Picamera2
import cv2

picam2 = Picamera2()
picam2.start()

frame = picam2.capture_array()
cv2.imwrite("frame.jpg", frame)

print("saved frame.jpg")