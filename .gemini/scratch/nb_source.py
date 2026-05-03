# --- Code Cell Execution Count: 1, Index: 0 ---
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
from skimage import measure

# --- Code Cell Execution Count: 4, Index: 1 ---
import cv2
import matplotlib.pyplot as plt

path = r'D:\Computer Vision Project\data\vietnamese_car_license_plate\test\CarLongPlate750_jpg.rf.9d25933f3b9e571485eb5e19eaf53b78.jpg'
original = cv2.imread(path)

# Chuyển đổi từ BGR sang RGB
original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

plt.imshow(original_rgb) 
plt.title('Original Image')
plt.axis('off')
plt.show()


# --- Code Cell Execution Count: 5, Index: 2 ---
gray_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

plt.imshow(gray_image, cmap='gray') 
plt.title('1. Gray Image')
plt.axis('off')
plt.show()

# --- Code Cell Execution Count: 6, Index: 3 ---
blur = cv2.GaussianBlur(gray_image.copy(), (5, 5), 0)

plt.imshow(blur, cmap='gray')
plt.title("2. Gaussian Blur")
plt.axis('off')
plt.show()

# --- Code Cell Execution Count: 7, Index: 4 ---
sobel = cv2.Sobel(blur, cv2.CV_8U, 1, 0, ksize=3)

plt.imshow(sobel, cmap='gray')
plt.title("3. Sobel X")
plt.axis('off')
plt.show()

# --- Code Cell Execution Count: 8, Index: 5 ---
_, thresh = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.imshow(thresh, cmap='gray')
plt.title("4. Otsu Threshold")
plt.axis('off')
plt.show()

# --- Code Cell Execution Count: 9, Index: 6 ---
kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)

plt.imshow(closed, cmap='gray')
plt.title("5. Morph Close")
plt.axis('off')
plt.show()

# --- Code Cell Execution Count: 10, Index: 7 ---
kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

plt.imshow(opened, cmap='gray')
plt.title("6. Morph Open")
plt.axis('off')
plt.show()

# --- Code Cell Execution Count: 11, Index: 8 ---
contours, _ = cv2.findContours(opened.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour_image = original.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2) 

plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
plt.title("7. Contours")
plt.axis('off')
plt.show()

# --- Code Cell Execution Count: 17, Index: 9 ---
original_copy = original.copy()
min_area = 3500
extracted_plates = []

for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    area = w * h
    aspect_ratio = w / float(h)
    
    if area > min_area and 3 <= aspect_ratio <= 7:
        cropped = original_copy[y:y+h, x:x+w]
        extracted_plates.append(cropped)
        if(len(extracted_plates) == 2):
            plt.figure()
            plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            plt.title(f"8. Extracted Plate Region {i + 1}")
            plt.axis('off')
            plt.show()

# --- Code Cell Execution Count: None, Index: 10 ---
def segment_chars(plate_img, fixed_width=400):
    
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    plate_img = imutils.resize(plate_img, width=fixed_width)
    thresh = imutils.resize(thresh, width=fixed_width)
    bgr_thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    labels = measure.label(thresh, background=0)
    charCandidates = np.zeros(thresh.shape, dtype='uint8')
    characters = []

    for label in np.unique(labels):
        if label == 0:
            continue
        labelMask = np.zeros(thresh.shape, dtype='uint8')
        labelMask[labels == label] = 255
        cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1] if imutils.is_cv3() else cnts[0]
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
            aspectRatio = boxW / float(boxH)
            solidity = cv2.contourArea(c) / float(boxW * boxH)
            heightRatio = boxH / float(plate_img.shape[0])
            if aspectRatio < 2.0 and solidity > 0.05 and 0.1 < heightRatio < 0.98 and boxW > 3:
                hull = cv2.convexHull(c)
                cv2.drawContours(charCandidates, [hull], -1, 255, -1)

    contours, _ = cv2.findContours(charCandidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0]) 
        addPixel = 4
        prev_x = -1
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            if y > addPixel:
                y -= addPixel
            else:
                y = 0
            if x > addPixel:
                x -= addPixel
            else:
                x = 0
            if prev_x != -1 and x - prev_x > w * 2:
                continue
            char_img = bgr_thresh[y:y + h + addPixel*2, x:x + w + addPixel*2]
            characters.append(char_img)
            prev_x = x + w
        return characters if 5 <= len(characters) <= 9 else None
    return None


for i, plate_img in enumerate(extracted_plates):
    chars = segment_chars(plate_img, fixed_width=400)
    if chars is not None:
        print(f"Plate has {len(chars)} characters detected.")
        for j, cimg in enumerate(chars):
            plt.figure()
            plt.imshow(cv2.cvtColor(cimg, cv2.COLOR_BGR2RGB))
            plt.title(f"Character {j+1} of Plate")
            plt.axis('off')
            plt.show()


