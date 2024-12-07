import cv2 as cv
from yolo_detection import YoloDetection
import pyfirmata2 as firmata
import time
import threading

# Inisialisasi Arduino
PORT = firmata.Arduino.AUTODETECT
board = firmata.Arduino(PORT)
print("Arduino terdeteksi!!!")
SERVO_PIN = 3
board.digital[SERVO_PIN].mode = firmata.SERVO

# Parameter YOLO
weights = 'yolov4-tiny.weights'
cfg = 'yolov4-tiny.cfg'
classes_file = 'classes.txt'

# Inisialisasi YOLO
yolo = YoloDetection(weights, cfg, classes_file)

# Video Capture
cap = cv.VideoCapture(0)

servo_state = False
def control_servo():
    global servo_state
    while True:
        if servo_state:
            print("TERDETEKSI MOBIL - BUKA SERVO")
            board.digital[SERVO_PIN].write(100)  # Servo terbuka
            time.sleep(5)  # Durasi membuka servo
            print("TERDETEKSI MOBIL - TUTUP SERVO")
            board.digital[SERVO_PIN].write(0)  # Servo tertutup
            time.sleep(1)  # Durasi menunggu sebelum deteksi berikutnya
            servo_state = False  # Setelah membuka, servo ditutup
        time.sleep(0.1)  # Delay singkat untuk thread kontrol servo

# Membuat thread untuk kontrol servo
servo_thread = threading.Thread(target=control_servo)
servo_thread.daemon = True  # Thread akan berhenti saat program utama selesai
servo_thread.start()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi objek pada frame
    classes, scores, boxes = yolo.detect_objects(frame)
    count = yolo.draw_detections(frame, classes, scores, boxes)

    if count >= 1 and not servo_state:
        servo_state = True

    # Hitung FPS
    fps = yolo.calculate_fps()

    # Tambahkan informasi ke frame
    cv.putText(frame, f'Jumlah Objek: {count}', (20, 50), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 2)
    cv.putText(frame, f'FPS: {fps:.2f}', (20, 80), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 2)

    # Tampilkan frame
    cv.imshow('frame', frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()