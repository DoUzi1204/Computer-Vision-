import os
from ultralytics import YOLO

def main():
    # --- CẤU HÌNH TRAIN ---
    # Đặt CONTINUE_TRAINING = True nếu bạn đã train xong 50 epoch và muốn train tiếp (ví dụ thêm 50 epoch nữa)
    CONTINUE_TRAINING = True
    
    if CONTINUE_TRAINING:
        print("Tiếp tục huấn luyện từ checkpoint tốt nhất đã có...")
        # Load lại model từ file weights tốt nhất của lần train trước
        model = YOLO(r'runs\detect\output\yolo11n_plate_detect-2\weights\best.pt')
    else:
        print("Khởi tạo mô hình YOLOv11 (bản nano) mới...")
        model = YOLO('yolo11n.pt')

    # Đường dẫn file data.yaml
    # Vui lòng điều chỉnh lại đường dẫn tuyệt đối cho chắc chắn nếu đường dẫn tương đối gặp lỗi
    data_path = r'D:\Computer Vision Project\data\Vietnamese Car Plate\data.yaml'

    # Thiết lập tham số huấn luyện
    # Nên dùng GPU (device=0) nếu máy có card rời NVIDIA
    print("Bắt đầu huấn luyện mô hình YOLOv11...")
    results = model.train(
        data=data_path,
        epochs=50,       # Số vòng huấn luyện, có thể tăng lên nếu muốn độ chính xác cao hơn
        imgsz=640,       # Kích thước ảnh đầu vào
        batch=16,        # Kích thước batch
        device='0',      # Thay '0' thành 'cpu' nếu máy bạn không có GPU NVIDIA
        project='output',# Thư mục lưu kết quả
        name='yolo11n_plate_detect', # Tên thư mục con lưu weights
        workers=0        # Fix lỗi WinError 1455 (hết memory/paging file)
    )
    print("Hoàn tất huấn luyện!")

if __name__ == '__main__':
    main()
