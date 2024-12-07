import cv2 as cv
import time


class YoloDetection:
    def __init__(self, weights, cfg, classes_file, conf_threshold=0.4, nms_threshold=0.4, input_size=(416, 416)):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.colors = [(255, 255, 255), (0, 234, 255), (255, 255, 0),
                       (255, 255, 0), (255, 0, 255), (0, 0, 255)]

        # Load class names
        with open(classes_file, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

        # Load YOLO model
        self.net = cv.dnn.readNet(weights, cfg)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        self.model = cv.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=input_size, scale=1 / 255, swapRB=True)

        # FPS tracking
        self.starting_time = time.time()
        self.frame_counter = 0

    def detect_objects(self, frame):
        try:
            classes, scores, boxes = self.model.detect(frame, self.conf_threshold, self.nms_threshold)
            classes = classes.flatten() if hasattr(classes, 'flatten') else classes
            scores = scores.flatten() if hasattr(scores, 'flatten') else scores
            return classes, scores, boxes
        except Exception as e:
            print(f"Error during detection: {e}")
            return [], [], []

    def draw_detections(self, frame, classes, scores, boxes):
        count = 0
        for classid, score, box in zip(classes, scores, boxes):
            color = self.colors[classid % len(self.colors)]
            label = f"{self.class_names[classid]}: {score:.2f}"
            object_name = self.class_names[classid]
            if object_name == "car":
                count += 1
            cv.rectangle(frame, box, color, 2)
            cv.putText(frame, label, (box[0], box[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return count

    def calculate_fps(self):
        self.frame_counter += 1
        elapsed_time = time.time() - self.starting_time
        fps = self.frame_counter / elapsed_time
        return fps
