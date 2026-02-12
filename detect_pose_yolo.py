import cv2
import numpy as np
from ultralytics import YOLO
from utils import pixel_to_world

MODEL_PATH = "yolov8n.pt"  
SCALE = 0.5  # мм на пиксель

model = YOLO(MODEL_PATH)

camera_matrix = np.load("camera_matrix.npy")
dist_coeffs = np.load("dist_coeffs.npy")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

    results = model(frame, verbose=False)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # угол через minAreaRect
            cropped = frame[int(y1):int(y2), int(x1):int(x2)]
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            angle = 0
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(cnt)
                angle = rect[2]

            # перевод в мм
            x_mm, y_mm = pixel_to_world((center_x, center_y), SCALE)

            # отрисовка
            cv2.rectangle(frame,
                          (int(x1), int(y1)),
                          (int(x2), int(y2)),
                          (0, 255, 0), 2)

            cv2.circle(frame,
                       (center_x, center_y),
                       5, (0, 0, 255), -1)

            cv2.putText(frame,
                        f"X: {x_mm:.1f} mm",
                        (center_x + 10, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 0, 0), 2)

            cv2.putText(frame,
                        f"Angle: {angle:.1f}",
                        (center_x + 10, center_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 0, 0), 2)

    cv2.imshow("YOLO Robot Vision", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
