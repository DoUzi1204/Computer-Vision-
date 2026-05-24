"""
char_segmentation_v2.py
-----------------------
Chứa hàm hỗ trợ phân đoạn (segment) ký tự trên ảnh biển số
sử dụng phương pháp thuần Vertical Projection Profile (VPP).
"""

import cv2
import numpy as np
import imutils

def segment_chars_vpp(plate_img, fixed_width=400):
    """
    Phân đoạn các ký tự trên ảnh biển số thuần bằng Vertical Projection Profile.
    
    LƯU Ý: 
    Phương pháp này không hỗ trợ tốt cho "biển số 2 dòng" (VD: biển xe máy).
    Vì nếu chiếu dọc trực tiếp, các chữ ở dòng trên và dòng dưới sẽ đè 
    lên cùng một cột (trục x) khiến thuật toán tưởng nhầm là 1 chữ duy nhất.

    Parameters
    ----------
    plate_img   : np.ndarray  – ảnh biển số (BGR)
    fixed_width : int         – chiều rộng resize trước khi segment

    Returns
    -------
    list[np.ndarray] | None
        Danh sách ảnh từng ký tự, hoặc None nếu không hợp lệ.
    """
    # 1. TIỀN XỬ LÝ
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 
        11, 2,
    )
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    plate_img = imutils.resize(plate_img, width=fixed_width)
    thresh = imutils.resize(thresh, width=fixed_width)
    bgr_thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    img_h, img_w = thresh.shape

    # 2. PURE VERTICAL PROJECTION (Không tách ngang)
    vertical_projection = np.sum(thresh, axis=0) / 255
    col_threshold = np.max(vertical_projection) * 0.35 
    is_text_col = vertical_projection > col_threshold
    
    characters = []
    in_col_segment = False
    start_x = 0
    
    def process_column(start_x, end_x):
        # Trích xuất dải dọc (chiều rộng = chữ, chiều cao = toàn bộ ảnh)
        col_thresh = thresh[:, start_x:end_x]
        
        # Tìm contour trong dải dọc để cắt gọt bớt râu ria ở trên/dưới
        cnts = cv2.findContours(col_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1] if imutils.is_cv3() else cnts[0]
        
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            (cx, cy, cw, ch) = cv2.boundingRect(c)
            
            # Tính lại tọa độ thực trên bức ảnh gốc
            final_x = start_x + cx
            final_y = cy
            final_w = cw
            final_h = ch
            
            aspect_ratio = final_w / float(final_h)
            height_ratio = final_h / float(img_h)
            
            # Lọc bằng kích thước/tỷ lệ
            if 0.1 < aspect_ratio < 1.5 and final_w > 5 and 0.25 < height_ratio < 0.95:
                addPixel = 4
                pad_x1 = max(0, final_x - addPixel)
                pad_x2 = min(img_w, final_x + final_w + addPixel)
                pad_y1 = max(0, final_y - addPixel)
                pad_y2 = min(img_h, final_y + final_h + addPixel)
                
                char_crop = bgr_thresh[pad_y1:pad_y2, pad_x1:pad_x2]
                characters.append((final_x, char_crop))

    for x, val in enumerate(is_text_col):
        if val and not in_col_segment:
            in_col_segment = True
            start_x = x
        elif not val and in_col_segment:
            in_col_segment = False
            process_column(start_x, x)
            
    if in_col_segment:
        process_column(start_x, len(is_text_col))

    if characters:
        # Sắp xếp các ký tự từ trái qua phải thuần túy theo trục x
        sorted_chars = sorted(characters, key=lambda item: item[0])
        final_crops = [crop for _, crop in sorted_chars]
        
        if 5 <= len(final_crops) <= 9:
            return final_crops

    return None
