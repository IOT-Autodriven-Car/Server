from transformers import (
    SegformerFeatureExtractor, 
    SegformerForSemanticSegmentation
)
from .config import VIS_LABEL_MAP as LABEL_COLORS_LIST
import os
from .utils import (
    draw_segmentation_map, 
    image_overlay,
    predict
)
from .extract_features import find_area_between_points_optimized

import cv2
import time

# Khởi tạo mô hình và extractor bên ngoài hàm để không phải tải lại mỗi khi gọi hàm.
extractor = SegformerFeatureExtractor()
current_directory = os.path.dirname(os.path.abspath(__file__))
model_iou_directory = os.path.join(current_directory, 'model_iou')

model = SegformerForSemanticSegmentation.from_pretrained(model_iou_directory)
device = 'cuda:0'
model.to(device).eval()

def inferface_frame(frame, imgsz=(400, 400)):
    """
    Hàm xử lý một frame và trả về các giá trị A, B, C, D, areaAB, areaAC.
    
    Args:
        frame: Một frame đầu vào (ảnh) dưới dạng numpy array.
        imgsz: Tuple (width, height) để resize ảnh đầu vào, mặc định là (400, 300).
    
    Returns:
        A, B, C, D, areaAB, areaAC: Kết quả từ hàm find_area_between_points_optimized.
    """
    # Resize frame nếu cần
    if imgsz is None:
        return None
    
    frame = cv2.resize(frame, (imgsz[0], imgsz[1]))
    
    # Chuyển đổi màu sắc từ BGR sang RGB (OpenCV mặc định là BGR)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Bắt đầu đếm thời gian xử lý
    start_time = time.time()

    # Sử dụng model để dự đoán nhãn
    labels = predict(model, extractor, frame_rgb, device)
    # print(labels.shape)
    
    # Tìm các tọa độ và diện tích giữa các điểm
    A, B, C, D, areaAB, areaAC = find_area_between_points_optimized(labels)

    
    # Tính FPS (Frame Per Second)
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    print(f"FPS: {fps}")

    # Get segmentation map.
    # seg_map = draw_segmentation_map(
    #     labels.cpu(), LABEL_COLORS_LIST
    # )
    # outputs = image_overlay(frame, seg_map)
    # cv2.imshow('Image', outputs)
    # cv2.waitKey(1)
    
    # Trả về các giá trị cần thiết
    return A, B, C, D, areaAB, areaAC
