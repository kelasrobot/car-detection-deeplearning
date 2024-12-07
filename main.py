import cv2 as cv
from yolo_detection import YoloDetection

# Parameter
weights = 'yolov4-tiny.weights'
cfg = 'yolov4-tiny.cfg'
classes_file = 'classes.txt'

# Inisialisasi YOLO
yolo = YoloDetection(weights, cfg, classes_file)

# Video Capture
cap = cv.VideoCapture("video.avi")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi objek
    classes, scores, boxes = yolo.detect_objects(frame)

    # Gambar deteksi pada frame
    count = yolo.draw_detections(frame, classes, scores, boxes)

    # Hitung FPS
    fps = yolo.calculate_fps()

    # Tambahkan informasi ke frame
    cv.putText(frame, f'Jumlah Objek: {count}', (20, 50), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 2)
    cv.putText(frame, f'FPS: {fps:.2f}', (20, 80), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 2)


    # Tampilkan frame
    cv.imshow('frame', frame)
    key = cv.waitKey(10)
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

