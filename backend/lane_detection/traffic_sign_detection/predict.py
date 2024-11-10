from ultralytics import YOLO
import cv2

from lane_detection.service.infer_frame import device
from lane_detection.traffic_sign_detection import model


def predict_on_frame(frame):
    """
    Predict frame and return labels and annotated frame
    Args:
        frame (np.ndarray): Hình ảnh đầu vào (frame từ video hoặc camera).
    Returns:
        labels (list): List of all label in the frame
        annotated_frame (np.ndarray): Img with bounding box and label
    """
    if frame.shape[0] != 640 or frame.shape[1] != 640:
        frame = cv2.resize(frame, (640, 640))

    result = model.predict(frame, imgsz=640, conf=0.5, device = "cpu")
    boxes = result[0].boxes  # Bounding boxes

    print("Boxes:", boxes)
    print("Labels:", [model.names[int(box.cls)] for box in boxes])

    annotated_frame = result[0].plot()

    # Return labels and the annotated frame
    labels = [model.names[int(box.cls)] for box in boxes]  # List of labels
    return labels, annotated_frame

