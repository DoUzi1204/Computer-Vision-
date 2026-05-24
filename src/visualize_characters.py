import os
import cv2
import matplotlib.pyplot as plt
import random

def visualize_character_dataset(dataset_path, output_path="dataset_visualization.png", grid_size=(5, 6)):
    """
    Hàm để tạo ra một ảnh trực quan hóa các lớp ký tự trong bộ dữ liệu.
    Lấy ngẫu nhiên 1 ảnh từ mỗi thư mục con (0-9, A-Z) và in ra dạng lưới (grid).
    
    Args:
        dataset_path (str): Đường dẫn tới thư mục chứa bộ dữ liệu ký tự.
        output_path (str): Đường dẫn lưu file ảnh kết quả.
        grid_size (tuple): Kích thước lưới (số hàng, số cột). Mặc định là (5, 6) cho 30 lớp.
    """
    if not os.path.exists(dataset_path):
        print(f"Không tìm thấy thư mục: {dataset_path}")
        return
        
    # Lấy danh sách các thư mục con và sắp xếp chúng (0-9, A-Z)
    folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    folders.sort()
    
    # Tạo figure cho matplotlib
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(12, 10))
    fig.suptitle('Trực quan hóa bộ dữ liệu ký tự', fontsize=20, fontweight='bold')
    
    # Làm phẳng mảng axes để dễ dàng lặp qua
    axes = axes.flatten()
    
    for idx, folder_name in enumerate(folders):
        if idx >= len(axes):
            break # Vượt quá số lượng ô trong lưới
            
        folder_path = os.path.join(dataset_path, folder_name)
        images = [img for img in os.listdir(folder_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
        
        if images:
            # Chọn ngẫu nhiên một ảnh trong thư mục
            random_image_name = random.choice(images)
            image_path = os.path.join(folder_path, random_image_name)
            
            # Đọc ảnh bằng OpenCV (Bỏ qua màu, đọc dạng Grayscale hoặc RGB tuỳ ý, ở đây dùng RGB để vẽ)
            img = cv2.imread(image_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[idx].imshow(img)
            else:
                axes[idx].text(0.5, 0.5, 'Error', ha='center', va='center')
        else:
            axes[idx].text(0.5, 0.5, 'No Image', ha='center', va='center')
            
        # Đặt tiêu đề cho từng ô là tên thư mục (ký tự)
        axes[idx].set_title(f"Class: {folder_name}", fontsize=14)
        axes[idx].axis('off')
        
    # Ẩn các ô trống nếu số lượng lớp ít hơn số lượng ô trong lưới
    for i in range(len(folders), len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Dataset visualization saved at: {output_path}")
    plt.show()

if __name__ == "__main__":
    DATASET_PATH = r"D:\Computer Vision Project\data\Character dataset"
    OUTPUT_IMAGE_PATH = r"D:\Computer Vision Project\data\character_dataset_visualization.png"
    
    visualize_character_dataset(DATASET_PATH, OUTPUT_IMAGE_PATH)
