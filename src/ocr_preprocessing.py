import cv2
import numpy as np

def preprocess_for_ocr(plate_img):
    """
    Tiền xử lý ảnh biển số (đã crop) trước khi đưa vào OCR 
    gồm đúng 4 bước: Deskew -> Khử Nhiễu -> Tăng Tương Phản -> Chuẩn hóa kích thước.
    """
    # ---------------------------------------------------------
    # BƯỚC 1: Deskew (Xoay Thẳng Góc)
    # ---------------------------------------------------------
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # Dùng thuật toán Hough Line Transform để tìm các đường thẳng trên biển số
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=30, maxLineGap=10)
    
    angle = 0.0
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Lọc chỉ lấy các đường gần ngang (nghiêng từ -30 đến 30 độ)
            if -30 < ang < 30:
                angles.append(ang)
        if len(angles) > 0:
            angle = np.median(angles) # Dùng trung vị để tránh nhiễu
            
    # Tiến hành xoay ảnh
    if abs(angle) > 0.5:
        (h, w) = plate_img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(plate_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    else:
        deskewed = plate_img.copy()

    # ---------------------------------------------------------
    # BƯỚC 2: Khử Nhiễu Hình Ảnh
    # ---------------------------------------------------------
    # Dùng Bilateral Filter để khử nhiễu, làm mịn màng nền nhưng vẫn giữ được độ sắc nét của viền chữ
    denoised = cv2.bilateralFilter(deskewed, d=11, sigmaColor=17, sigmaSpace=17)

    # ---------------------------------------------------------
    # BƯỚC 3: Tăng Cường Tương Phản
    # ---------------------------------------------------------
    # Sử dụng thuật toán CLAHE trên không gian màu LAB để tăng tương phản mảng sáng tối
    # mà không làm hỏng màu tự nhiên của bức ảnh
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # ---------------------------------------------------------
    # BƯỚC 4: Chuẩn Hóa Kích Thước
    # ---------------------------------------------------------
    # PaddleOCR cấu hình nhận ảnh chiều cao 48px, do đó ta chuẩn hóa chiều cao ảnh về 48,
    # chiều rộng scale theo tỷ lệ tương ứng để không làm méo chữ.
    h, w = enhanced.shape[:2]
    target_h = 48
    target_w = int(w * (target_h / float(h)))
    if target_w < target_h: 
        target_w = target_h
        
    final_img = cv2.resize(enhanced, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    return final_img
