# Nhận diện biển số xe (License Plate Recognition)

Dự án nhận diện biển số xe ứng dụng các kỹ thuật Xử lý ảnh (Computer Vision) và Học sâu (Deep Learning) để phát hiện và nhận dạng ký tự trên biển số ô tô tại Việt Nam.

## 1. Phân chia công việc

| Họ và tên            | MSV      | Công việc                                                                                                                                          |
| :------------------- | :------- | :------------------------------------------------------------------------------------------------------------------------------------------------- |
| Phạm Quý Đô (Leader) | 22001562 | - Lọc dữ liệu<br>- Phát hiện biển số bằng YOLOv11<br>- Phân đoạn ký tự bằng Vertical Projection vs Connected Components<br>- Xây dựng GUI hệ thống |
| Nguyễn Đình Duy      | 22001554 | - Tiền xử lí<br>- Phát hiện vùng biển số bằng Edge + Morphology<br>- Fine-tune PaddleOCR                                                           |
| Lê Tuấn Hiệp         | 22001577 | - Tiền xử lí<br>- Phát hiện vùng biển số bằng Contour based<br>- Tiền xử lí chuẩn bị OCR                                                           |

## 2. Bộ dữ liệu sử dụng

Dự án sử dụng các bộ dữ liệu chính được lấy từ nền tảng Kaggle và Roboflow:

- **Bộ dữ liệu biển số xe (Vietnamese Car License Plate):** Sử dụng để kiểm thử thuật toán phát hiện vùng biển số.<br>
  🔗 [Link Kaggle](https://www.kaggle.com/datasets/duynguyn3/vietnam-car-license-plate)
- **Bộ dữ liệu ký tự biển số xe (Character dataset for Vietnam license plate):** Sử dụng để huấn luyện mô hình CNN nhận diện từng ký tự được cắt ra từ biển số.<br>
  🔗 [Link Kaggle](https://www.kaggle.com/datasets/nguyenquanglinh0109/character-dataset-for-vietnam-license-plate)
- **Bộ dữ liệu Vietnamese License Plate OCR:** Sử dụng để fine-tune mô hình PaddleOCR.<br>
  🔗 [Link Kaggle](https://www.kaggle.com/datasets/wirqhuy/vietnamese-license-plate-ocr)
- **Bộ dữ liệu Vietnamese Car License Plate (Roboflow):** Sử dụng để huấn luyện mô hình YOLOv11.<br>
  🔗 [Link Roboflow](https://universe.roboflow.com/cuong-ta-ulxex/vietnamese-car-license-plate/dataset/1)

## 3. Cách tổ chức các thư mục

Cấu trúc thư mục của dự án được tổ chức như sau:

```text
Computer Vision Project/
│
├── data/                       # Chứa dữ liệu ảnh biển số xe và ảnh ký tự
│   ├── Character dataset/      # Dữ liệu ảnh các ký tự (A-Z, 0-9)
│   ├── vietnamese car license plate/ # Dữ liệu ảnh các phương tiện chứa biển số
│   ├── Vietnamese Car Plate/   # Dữ liệu huấn luyện YOLOv11 (từ Roboflow)
│   └── License Plate OCR/      # Dữ liệu fine-tune PaddleOCR
│
├── models/                     # Thư mục lưu trữ các mô hình đã được huấn luyện (CNN, YOLO)
│
├── notebooks/                  # Chứa các file Jupyter Notebook dùng để thực nghiệm thuật toán
│   └── thucnghiem.ipynb        # Notebook chạy thử nghiệm các bước xử lý ảnh và nhận diện
│
├── PaddleOCR/                  # Thư viện mã nguồn mở PaddleOCR clone về
│
├── src/                        # Chứa các mã nguồn (source code) Python chính của dự án
│   ├── app.py                  # Mã nguồn chính chạy giao diện người dùng (GUI)
│   ├── yolo_detection.py       # Module phát hiện biển số bằng mô hình YOLOv11
│   ├── ocr_preprocessing.py    # Module tiền xử lý ảnh trước khi gọi OCR
│   ├── db_manager.py           # Module quản lý cơ sở dữ liệu lưu trữ kết quả (SQLite)
│   ├── Contour_Filtering.py    # Phát hiện biển số dựa trên Contour
│   ├── Edge_Morphology.py      # Phát hiện biển số dựa trên Edge & Morphology
│   ├── char_segmentation.py    # Cắt / phân đoạn ký tự từ vùng biển số
│   ├── character_recognition.py# Nhận diện ký tự bằng mô hình học máy / học sâu
│   ├── train_yolov11.py        # Code huấn luyện mô hình YOLOv11
│   ├── train_cnn.py            # Code huấn luyện mô hình CNN nhận diện ký tự
│   └── ...
│
├── output/                     # Thư mục lưu kết quả ảnh đầu ra, database (parking.db)
│
├── runs/                       # Thư mục chứa các log, đồ thị huấn luyện của YOLO
│
├── vi_plate_rec.yml            # File cấu hình fine-tune PaddleOCR
├── requirements.txt            # Danh sách các thư viện Python cần cài đặt
├── .gitignore                  # File cấu hình bỏ qua log/models/data trên Git
└── README.md                   # File tài liệu hướng dẫn (chính là file này)
```

## 4. Kịch bản thực nghiệm

Dự án được tiến hành thực nghiệm qua các bước chính nhằm so sánh hiệu năng giữa các phương pháp xử lý ảnh truyền thống và học sâu (Deep Learning):

### 4.1. Thực nghiệm phát hiện vùng biển số (License Plate Detection)

- **Phương pháp truyền thống:** Sử dụng **Contour-based** và kết hợp cạnh với hình thái học (**Edge + Morphology**). Trích xuất và lọc các vùng dựa trên tỷ lệ, kích thước đặc trưng của biển số.
- **Phương pháp Deep Learning:** Huấn luyện và sử dụng mô hình **YOLOv11** trên bộ dữ liệu từ Roboflow để tự động định vị (bounding box) vùng biển số một cách chính xác.
- **Mục tiêu:** Đánh giá và so sánh độ chính xác (mAP, Precision, Recall) cũng như tốc độ (FPS) giữa YOLOv11 và các phương pháp xử lý ảnh truyền thống.

### 4.2. Thực nghiệm phân đoạn ký tự (Character Segmentation)

- **Tiền xử lý:** Chuyển đổi ảnh xám, làm mờ giảm nhiễu và nhị phân hóa (Thresholding) vùng biển số thu được.
- **Phân đoạn:** Triển khai và đánh giá hai phương pháp độc lập để cô lập và cắt rời từng ký tự:
  - Phương pháp 1: Sử dụng thuật toán chiếu đứng (**Vertical Projection**).
  - Phương pháp 2: Phân tích các thành phần liên thông (**Connected Components**).
- **Xử lý nhiễu:** Loại bỏ các nhiễu rác (như đinh ốc, viền) thông qua ngưỡng diện tích/chiều cao, sau đó sắp xếp các ký tự theo đúng thứ tự trên biển vuông hoặc biển dài.

### 4.3. Thực nghiệm nhận diện ký tự (Character Recognition)

- **Phương pháp Mạng tự xây dựng:** Xây dựng và huấn luyện mô hình **CNN** để phân loại các ký tự (0-9 và A-Z) trên bộ dữ liệu `Character dataset`.
- **Phương pháp Fine-tune OCR:** Tinh chỉnh mô hình **PaddleOCR** bằng bộ dữ liệu `Vietnamese License Plate OCR` để nâng cao khả năng nhận dạng trên các ảnh thực tế phức tạp.
- **Mục tiêu:** So sánh sự ổn định và độ chính xác của mô hình CNN cơ bản so với hệ thống OCR chuyên dụng. Cuối cùng, tích hợp phương pháp tối ưu nhất vào **Giao diện người dùng (GUI)** hoàn chỉnh của hệ thống.

## 5. Thiết lập môi trường & Chạy thử

### 5.1. Thiết lập môi trường

Khuyến nghị sử dụng môi trường ảo (`.venv` hoặc `conda`) để tránh xung đột thư viện.

**Bước 1:** Mở terminal/command prompt tại thư mục gốc của dự án (`d:\Computer Vision Project`).
**Bước 2:** Kích hoạt môi trường ảo (Dự án đã có sẵn thư mục `.venv`):

```bash
# Kích hoạt trên Windows
.venv\Scripts\activate
```

_(Nếu môi trường chưa có sẵn hoặc bị lỗi, bạn có thể tạo mới bằng lệnh `python -m venv .venv`)_

**Bước 3:** Cài đặt các thư viện cần thiết bằng lệnh:

```bash
pip install -r requirements.txt
```

### 5.2. Chạy thử chương trình

Dự án hỗ trợ nhiều phương pháp chạy thử, từ ứng dụng GUI hoàn chỉnh đến các file script thực nghiệm:

**Cách 1: Chạy Giao diện người dùng (GUI) - Khuyến nghị**
Đây là hệ thống cuối cùng đã tích hợp YOLOv11, hệ thống nhận diện ký tự và các phương pháp xử lý tối ưu nhất.

```bash
python src/app.py
```

_(Giao diện ứng dụng sẽ hiển thị, cho phép bạn tải ảnh chụp xe lên, tự động phát hiện, cắt biển số, nhận diện chữ số và lưu kết quả)._

**Cách 2: Chạy các script đánh giá độc lập (Terminal)**
Nếu bạn muốn đánh giá hoặc đo lường độ chính xác của từng module riêng lẻ:

- Chạy script đánh giá độ chính xác phát hiện vùng biển số (YOLOv11):
  ```bash
  python src/evaluate_detection.py
  ```
- Chạy script đánh giá độ chính xác nhận diện ký tự:
  ```bash
  python evaluate_recognition.py
  ```

**Cách 3: Theo dõi quá trình xử lý qua Jupyter Notebook (Dành cho nghiên cứu)**
Mở file `thucnghiem.ipynb` để chạy và trực quan hóa từng bước thuật toán (tiền xử lý, lọc nhiễu, cắt ký tự bằng Vertical Projection, v.v.):

```bash
jupyter notebook notebooks/thucnghiem.ipynb
```
