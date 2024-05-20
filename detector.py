import cv2
import math
from ultralytics import YOLO

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
model = YOLO(r'C:\Users\Lenovo\Desktop\MIPT\DZ_2_SEM\ML\yolov8n.pt')

while True:
    success, img = cap.read()
    results = model(img, stream=True, save=True)
    for r in results:
        boxes = r.boxes
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            cv2.putText(img, model.names.get(box.cls.item()), org, font, fontScale, color, thickness)
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break