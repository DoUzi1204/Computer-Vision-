"""
yolo_detection.py
------------------
Chứa class YOLOPlateFinder – phát hiện vùng biển số xe sử dụng mô hình YOLOv11.

Phụ thuộc:
    char_segmentation.segment_chars  – phân đoạn ký tự trên biển số
"""

import cv2
import numpy as np
from ultralytics import YOLO
from char_segmentation import segment_chars

class YOLOPlateFinder:
    """
    Phát hiện biển số xe bằng mô hình YOLOv11.
    Tương thích với interface của PlateFinder (sử dụng OpenCV).
    """

    def __init__(self, model_path: str):
        """
        Khởi tạo mô hình YOLO.
        :param model_path: Đường dẫn tới file weights đã train (ví dụ: best.pt)
        """
        # Load mô hình
        self.model = YOLO(model_path)
        self.char_on_plate = []
        self.corresponding_area = []

    def find_possible_plates(self, input_img: np.ndarray, plate_label=None, conf_thresh=0.25):
        """
        Tìm tất cả biển số trong ảnh đầu vào bằng YOLO.

        Returns
        -------
        list[np.ndarray] | None  – danh sách ảnh biển số cắt ra (crop) được
        """
        self.char_on_plate = []
        self.corresponding_area = []
        plates = []

        # Chạy inference
        results = self.model(input_img, conf=conf_thresh, verbose=False)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Lấy toạ độ bounding box: x_min, y_min, x_max, y_max
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Cắt ảnh vùng biển số
                # Cần đảm bảo toạ độ không vượt quá kích thước ảnh
                h_img, w_img = input_img.shape[:2]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w_img, x2)
                y2 = min(h_img, y2)
                
                plate_roi = input_img[y1:y2, x1:x2]
                if plate_roi.size == 0:
                    continue

                plates.append(plate_roi)
                
                # Lưu toạ độ dạng [x, y, w, h] giống OpenCV cv2.boundingRect
                w = x2 - x1
                h = y2 - y1
                self.corresponding_area.append([x1, y1, w, h])

        return plates if plates else None
