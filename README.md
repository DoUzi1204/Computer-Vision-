# Nhận diện biển số xe (License Plate Recognition)

Dự án nhận diện biển số xe ứng dụng các kỹ thuật Xử lý ảnh (Computer Vision) và Học sâu (Deep Learning) để phát hiện và nhận dạng ký tự trên biển số ô tô tại Việt Nam.

## 1. Phân chia công việc

| Họ và tên            | MSV      | Công việc                                                                |
| :------------------- | :------- | :----------------------------------------------------------------------- |
| Phạm Quý Đô (Leader) | 22001562 | - Lọc dữ liệu<br>- Phân đoạn ký tự biển số<br>- Nhận diện kí tự bằng CNN |
| Nguyễn Đình Duy      | 22001554 | - Tiền xử lí<br>- Phát hiện vùng biển số bằng Edge + Morphology          |
| Lê Tuấn Hiệp         | 22001577 | - Tiền xử lí<br>- Phát hiện vùng biển số bằng Contour based              |

## 2. Bộ dữ liệu sử dụng

Dự án sử dụng 2 bộ dữ liệu chính được lấy từ nền tảng Kaggle:

- **Bộ dữ liệu biển số xe (Vietnamese Car License Plate):** Sử dụng để huấn luyện và kiểm thử thuật toán phát hiện vùng biển số.<br>
  🔗 [Link Kaggle](https://www.kaggle.com/datasets/duynguyn3/vietnam-car-license-plate)
- **Bộ dữ liệu ký tự biển số xe (Character dataset for Vietnam license plate):** Sử dụng để huấn luyện mô hình CNN nhận diện từng ký tự được cắt ra từ biển số.<br>
  🔗 [Link Kaggle](https://www.kaggle.com/datasets/nguyenquanglinh0109/character-dataset-for-vietnam-license-plate)

## 3. Cách tổ chức các thư mục

Cấu trúc thư mục của dự án được tổ chức như sau:

```text
Computer Vision Project/
│
├── data/                       # Chứa dữ liệu ảnh biển số xe và ảnh ký tự
│   ├── Character dataset/      # Dữ liệu ảnh các ký tự (A-Z, 0-9)
│   └── vietnamese car license plate/ # Dữ liệu ảnh các phương tiện chứa biển số
│
├── models/                     # Thư mục lưu trữ các mô hình đã được huấn luyện (VD: CNN weights)
│
├── notebooks/                  # Chứa các file Jupyter Notebook dùng để thực nghiệm thuật toán
│   └── thucnghiem.ipynb        # Notebook chạy thử nghiệm các bước xử lý ảnh và nhận diện
│
├── src/                        # Chứa các mã nguồn (source code) Python chính của dự án
│   ├── Contour_Filtering.py    # Phát hiện biển số dựa trên Contour
│   ├── Edge_Morphology.py      # Phát hiện biển số dựa trên Edge & Morphology
│   ├── char_segmentation.py    # Cắt / phân đoạn ký tự từ vùng biển số
│   ├── character_recognition.py# Nhận diện ký tự bằng mô hình học máy / học sâu
│   ├── train_cnn.py            # Code huấn luyện mô hình CNN nhận diện ký tự
│   └── ...
│
├── output/                     # Thư mục lưu kết quả ảnh đầu ra sau khi xử lý/nhận diện
│
├── requirements.txt            # Danh sách các thư viện Python cần cài đặt
└── README.md                   # File tài liệu hướng dẫn (chính là file này)
```

## 4. Thiết lập môi trường & Chạy thử

### 4.1. Thiết lập môi trường

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

### 4.2. Chạy thử chương trình

Có hai cách để chạy thử và kiểm tra các chức năng của dự án:

**Cách 1: Chạy trực tiếp mã nguồn bằng Python script**
Mở terminal và chạy thử các file mã nguồn độc lập trong thư mục `src`:

- Ví dụ, để chạy file test xử lý ảnh:
  ```bash
  python src/test.py
  ```

**Cách 2: Chạy trên Jupyter Notebook (Khuyến nghị để xem từng bước xử lý)**
Mở file `thucnghiem.ipynb` nằm trong thư mục `notebooks` để chạy lần lượt từng cell. Bạn có thể dễ dàng theo dõi hình ảnh đầu ra sau mỗi bước tiền xử lý, tìm biển số, và phân đoạn ký tự:

```bash
jupyter notebook notebooks/thucnghiem.ipynb
```
