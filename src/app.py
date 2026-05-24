import streamlit as st
import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
import pandas as pd
import sqlite3
import os
import sys

# Thêm đường dẫn thư mục hiện tại để có thể import các module bên trong src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from yolo_detection import YOLOPlateFinder
from db_manager import init_db, log_vehicle, DB_PATH
from ocr_preprocessing import preprocess_for_ocr

# --- KHỞI TẠO CACHE MÔ HÌNH ---
@st.cache_resource
def load_models():
    """Tải và lưu trữ các mô hình AI vào bộ nhớ tạm (Cache) để chạy nhanh hơn"""
    yolo_model_path = r'runs\detect\output\yolo11n_plate_detect-2\weights\best.pt'
    try:
        yolo_finder = YOLOPlateFinder(model_path=yolo_model_path)
    except Exception as e:
        yolo_finder = None
        st.error(f"Không thể tải mô hình YOLO. Vui lòng kiểm tra lại đường dẫn: {yolo_model_path}")
    
    try:
        ocr_reader = PaddleOCR(
            use_gpu=False,
            rec_model_dir=os.path.join(os.path.dirname(__file__), '../models/inference/rec_vi_plate'),
            rec_char_dict_path=os.path.join(os.path.dirname(__file__), '../data/License Plate OCR/vi_plate_dict.txt'),
            rec_algorithm='SVTR_LCNet',
            rec_image_shape='3, 48, 320',
            use_space_char=False,
            det=False,
            cls=False,
            show_log=False
        )
    except Exception as e:
        ocr_reader = None
        st.error(f"Không thể tải mô hình OCR: {str(e)}")
        
    return yolo_finder, ocr_reader

# Khởi tạo Database
init_db()

st.set_page_config(page_title="Hệ thống Trông Giữ Xe Thông Minh", layout="wide", page_icon="🚗")

# --- HÀM XỬ LÝ ẢNH ---
def process_plate_image(image, yolo_finder, ocr_reader):
    """Sử dụng YOLO để cắt biển số và EasyOCR để đọc chữ"""
    if yolo_finder is None:
        return None, None

    # Chuyển đổi ảnh sang định dạng OpenCV
    img_cv = np.array(image)
    if img_cv.shape[-1] == 4: # RGBA -> BGR
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2BGR)
    elif len(img_cv.shape) == 3: # RGB -> BGR
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    possible_plates = yolo_finder.find_possible_plates(img_cv)
    
    if possible_plates and len(possible_plates) > 0:
        # Lấy biển số đầu tiên tìm thấy
        plate_roi = possible_plates[0]
        
        # Tiền xử lý 4 bước
        ocr_ready_crop = preprocess_for_ocr(plate_roi)
        
        # Đọc chữ bằng PaddleOCR
        ocr_results = ocr_reader.ocr(ocr_ready_crop, det=False, cls=False)
        if ocr_results and len(ocr_results) > 0 and len(ocr_results[0]) > 0:
            recognized_plate = ocr_results[0][0][0]
        else:
            recognized_plate = ""
        
        # Chuyển đổi màu ảnh biển số lại thành RGB để hiển thị trên web
        plate_roi_rgb = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2RGB)
        return plate_roi_rgb, recognized_plate
    
    return None, None


# --- GIAO DIỆN CHÍNH ---
yolo_finder, ocr_reader = load_models()

st.title("🚗 Hệ Thống Trông Giữ Xe Thông Minh")
st.markdown("---")

tab1, tab2 = st.tabs(["📸 Quét Xe (Check-IN/OUT)", "📊 Quản lý Lãi Bãi Xe"])

with tab1:
    st.header("Quét biển số xe vào/ra")
    st.write("Vui lòng tải bức ảnh chụp biển số xe tại cổng để hệ thống tự động ghi nhận.")
    
    uploaded_file = st.file_uploader("Tải ảnh lên...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Hiển thị ảnh vừa tải lên
        image = Image.open(uploaded_file)
        st.image(image, caption='Ảnh chụp tại cổng', width=400)
        
        if st.button("🔍 Tiến hành Quét Biển Số", use_container_width=True):
            with st.spinner("Đang sử dụng AI để nhận diện biển số..."):
                plate_img, plate_text = process_plate_image(image, yolo_finder, ocr_reader)
                
            if plate_img is not None and plate_text:
                st.success("Nhận diện thành công!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(plate_img, caption=f"Vùng Biển Số (Crop)", width=200)
                with col2:
                    st.metric(label="Biển số xe nhận diện được", value=plate_text)
                    
                    # Logic Check-IN / Check-OUT
                    status, log_time = log_vehicle(plate_text)
                    if status == "IN":
                        st.info(f"✅ Đã ghi nhận **VÀO BÃI** lúc: {log_time}")
                    else:
                        st.warning(f"🛫 Đã ghi nhận **RA KHỎI BÃI** lúc: {log_time}")
                        
            else:
                st.error("Không tìm thấy biển số xe nào trong ảnh hoặc mô hình chưa quét được!")


with tab2:
    st.header("Danh sách phương tiện")
    
    # Đọc dữ liệu từ SQLite
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM parking_history ORDER BY id DESC", conn)
    conn.close()
    
    if df.empty:
        st.write("Bãi đỗ xe hiện đang trống, chưa có lịch sử.")
    else:
        # Tính toán sơ bộ
        total_in = len(df[df['status'] == 'IN'])
        
        # Thẻ thông tin nhanh
        st.metric(label="🚗 Số lượng xe ĐANG CÓ TRONG BÃI", value=total_in)
        
        st.markdown("### 📋 Lịch sử Ra / Vào")
        # Format lại bảng
        df_display = df.copy()
        df_display.columns = ["ID", "Biển số xe", "Thời gian VÀO", "Thời gian RA", "Trạng thái"]
        st.dataframe(df_display, use_container_width=True)
