"""Обработка изображений."""

import cv2
import numpy as np
import os
import time

image_id = "73370"
input_path = f"paintings/{image_id}.jpg"
output_dir = "paintings"

def halftone_(img):
    gray = (0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0])
    return np.clip(gray, 0, 255).astype(np.uint8)

def svertka_2(image, kernel):
    if len(image.shape) == 3:
        h, w, c = image.shape
        kh, kw = kernel.shape
        h_pad, w_pad = kh // 2, kw // 2

        padded = np.pad(image,((h_pad, h_pad), (w_pad, w_pad), (0, 0)),mode='constant')
        result = np.zeros((h, w, c))

        for y in range(h):
            for x in range(w):
                for channel in range(c):
                    region = padded[y:y + kh, x:x + kw, channel]
                    result[y, x, channel] = np.sum(region * kernel)
        return result
    else:
        h, w = image.shape
        kh, kw = kernel.shape
        h_pad, w_pad = kh // 2, kw // 2

        padded = np.pad(image, ((h_pad, h_pad), (w_pad, w_pad)), mode='constant')
        result = np.zeros((h, w), dtype=np.float32)

        for y in range(h):
            for x in range(w):
                region = padded[y:y + kh, x:x + kw]
                result[y, x] = np.sum(region * kernel)
        return result
def svertka_(image, kernel):
    if len(image.shape) == 2:
        h, w = image.shape
        kh, kw = kernel.shape
        h_pad, w_pad = kh // 2, kw // 2
        padded = np.pad(image, ((h_pad, h_pad), (w_pad, w_pad)), mode='constant')
        result = np.zeros((h, w))
        for y in range(h):
            for x in range(w):
                region = padded[y:y + kh, x:x + kw]
                result[y, x] = np.sum(region * kernel)
        return result
    elif len(image.shape) == 3:
        h, w, c = image.shape
        result = np.zeros((h, w, c))
        for channel in range(c):
            single_channel = image[:, :, channel]
            result[:, :, channel] = svertka_(single_channel, kernel)
        return result
def gauss_(image, size=5, sigma=1.0):
    center = size // 2
    x, y = np.meshgrid(np.arange(size) - center, np.arange(size) - center)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return svertka_(image, kernel)

def sobel_(image):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    if len(image.shape) == 3:
        gray_image = halftone_(image)
        grad_x = svertka_(gray_image, kernel_x)
        grad_y = svertka_(gray_image, kernel_y)
    else:
        grad_x = svertka_(image, kernel_x)
        grad_y = svertka_(image, kernel_y)

    result = np.sqrt(grad_x ** 2 + grad_y ** 2)

    if result.max() > 0:
        result = (result / result.max() * 255).astype(np.uint8)

    return result


print("ПРОВЕРКА РАБОТЫ С ИСПОЛЬЗОВАНИЕМ OpenCV")
# Загружаем изображение
img = cv2.imread(input_path)
print("\n1. cv2.cvtColor()")
start = time.time()
halftone_my = halftone_(img)
my_time = time.time() - start
print(f"Ручная: {my_time:.6f} сек")

start = time.time()
halftone_cv2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv_time = time.time() - start
print(f"cv2.cvtColor: {cv_time:.6f} сек")

cv2.imwrite(f"{output_dir}/{image_id}_halftone_my.jpg", halftone_my)
cv2.imwrite(f"{output_dir}/{image_id}_halftone_cv2.jpg", halftone_cv2)

print("\n2. cv2.filter2D() - размытие Гаусса")
start = time.time()
gauss_my = gauss_(img)
gauss_my = np.clip(gauss_my, 0, 255).astype(np.uint8)
my_time = time.time() - start
print(f"Ручная: {my_time:.6f} сек")

size, sigma = 5, 1.0
center = size // 2
x, y = np.meshgrid(np.arange(size) - center, np.arange(size) - center)
gauss_kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
gauss_kernel = gauss_kernel / gauss_kernel.sum()

start = time.time()
gauss_filter2d = cv2.filter2D(img, -1, gauss_kernel)
filter2d_time = time.time() - start
print(f"cv2.filter2D: {filter2d_time:.6f} сек")

cv2.imwrite(f"{output_dir}/{image_id}_gauss_my.jpg", gauss_my)
cv2.imwrite(f"{output_dir}/{image_id}_gauss_cv2.jpg", gauss_filter2d)

print("\n3. cv2.filter2D() - оператор Собеля")
start = time.time()
sobel_my = sobel_(img)
my_time = time.time() - start
print(f"Ручная: {my_time:.6f} сек")

kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
start = time.time()
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sobelx = cv2.filter2D(gray_img, cv2.CV_32F, kernel_x)
sobely = cv2.filter2D(gray_img, cv2.CV_32F, kernel_y)
sobel_filter2d = np.sqrt(sobelx ** 2 + sobely ** 2)
sobel_filter2d = (sobel_filter2d / sobel_filter2d.max() * 255).astype(np.uint8)
filter2d_time = time.time() - start
print(f"cv2.filter2D: {filter2d_time:.6f} сек")

cv2.imwrite(f"{output_dir}/{image_id}_sobel_my.jpg", sobel_my)
cv2.imwrite(f"{output_dir}/{image_id}_sobel_cv2.jpg", sobel_filter2d)

print("\n4. cv2.Canny()")
print("-" * 40)

start = time.time()
canny_result = cv2.Canny(halftone_my, 50, 150)
canny_time = time.time() - start
print(f"cv2.Canny: {canny_time:.6f} сек")

cv2.imwrite(f"{output_dir}/{image_id}_canny.jpg", canny_result)

print("\n5. cv2.cornerHarris()")
print("-" * 40)

start = time.time()
gray_float = np.float32(halftone_my)
harris = cv2.cornerHarris(gray_float, 2, 3, 0.04)
harris = cv2.dilate(harris, None)
img_harris = img.copy()
img_harris[harris > 0.01 * harris.max()] = [0, 0, 255]
harris_time = time.time() - start
print(f"cv2.cornerHarris: {harris_time:.6f} сек")

corners_count = np.sum(harris > 0.01 * harris.max())
print(f"Найдено углов: {corners_count}")

cv2.imwrite(f"{output_dir}/{image_id}_harris.jpg", img_harris)

"# New comment" 
