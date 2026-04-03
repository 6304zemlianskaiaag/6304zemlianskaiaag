
import cv2
import csv
import os
import random
import requests
import numpy as np
import time
from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass, field
import functools
from abc import ABC, abstractmethod


def time_method(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f" Время выполнения {func.__name__}: {end - start:.2f} сек.")
        return result

    return wrapper



class Artwork(ABC):
    __slots__ = ('__img', '__metadata', '__kernel', '__object_id', '__image_url')

    def __init__(self) -> None:
        self.__img = None
        self.__metadata = None
        self.__kernel = None
        self.__object_id = None
        self.__image_url = None
    @abstractmethod
    def halftone_(self) -> np.ndarray:
        pass
    @abstractmethod
    def svertka_(self, kernel: Optional[np.ndarray] = None) -> np.ndarray:
        pass
    @abstractmethod
    def gauss_(self, size: int = 5, sigma: float = 1.0) -> np.ndarray:
        pass
    @abstractmethod
    def sobel_(self) -> np.ndarray:
        pass

    @property
    def img(self) -> Optional[np.ndarray]:
        return self.__img
    @img.setter
    def img(self, val: Optional[np.ndarray]) -> None:
        if val is not None and not isinstance(val, np.ndarray):
            raise TypeError("Ошибка!!!")
        self.__img = val
    @property
    def metadata(self) -> Optional[Dict[str, Any]]:
        return self.__metadata
    @metadata.setter
    def metadata(self, value: Optional[Dict[str, Any]]) -> None:
        self.__metadata = value
    @property
    def kernel(self) -> Optional[np.ndarray]:
        if self.__kernel is not None:
            return self.__kernel
        raise ValueError('Ошибка!!!')
    @kernel.setter
    def kernel(self, matrix: np.ndarray) -> None:
        h, w = matrix.shape
        if h == w:
            self.__kernel = matrix
        else:
            raise ValueError('Ошибка!!!')
    @property
    def object_id(self) -> Optional[str]:
        return self.__object_id
    @object_id.setter
    def object_id(self, value: str) -> None:
        self.__object_id = value
    def __add__(self, other: Union['Artwork', int, float]) -> 'Artwork':
        if isinstance(other, (int, float)):
            result = np.clip(self.img.astype(np.int16) + other, 0, 255).astype(np.uint8)
            new_object = self.__class__()
            new_object.img = result
            return new_object

        if isinstance(other, Artwork):
            h = max(self.img.shape[0], other.img.shape[0])
            w = max(self.img.shape[1], other.img.shape[1])

            if len(self.img.shape) == 3:
                c1 = 3
            else:
                c1 = 1
            if len(other.img.shape) == 3:
                c2 = 3
            else:
                c2 = 1
            c = max(c1, c2)

            board = np.zeros((h, w, c), dtype=np.float32)

            img1 = self.img.astype(np.float32)
            img2 = other.img.astype(np.float32)

            if img1.ndim == 2:
                img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            if img2.ndim == 2:
                img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

            board[:self.img.shape[0], :self.img.shape[1]] += img1
            board[:other.img.shape[0], :other.img.shape[1]] += img2
            result = np.clip(board, 0, 255).astype(np.uint8)

            new_object = self.__class__()
            new_object.img = result
            return new_object

    def __radd__(self, other: Union[int, float]) -> 'Artwork':
        return self.__add__(other)

    def __str__(self) -> str:
        return f"{self.__class__.__name__} (ID: {self.object_id}, размер: {self.img.shape})"



class GrayscaleArtwork(Artwork):
    __slots__ = ('_image_type',)
    def __init__(self, image: Optional[np.ndarray] = None) -> None:
        super().__init__()
        if image is not None:
            if len(image.shape) == 3:
                self.img = self._to_grayscale(image)
            else:
                self.img = image.copy()
        self._image_type = "grayscale"

    def _to_grayscale(self, img: np.ndarray) -> np.ndarray:
        gray = (0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0])
        return np.clip(gray, 0, 255).astype(np.uint8)

    def halftone_(self) -> np.ndarray:
        return self.img.copy()

    def svertka_(self, kernel: Optional[np.ndarray] = None) -> np.ndarray:
        if len(self.img.shape) == 2:
            h, w = self.img.shape
            kh, kw = kernel.shape
            h_pad, w_pad = kh // 2, kw // 2
            padded = np.pad(self.img, ((h_pad, h_pad), (w_pad, w_pad)), mode='constant')
            result = np.zeros((h, w), dtype=np.float32)
            for y in range(h):
                for x in range(w):
                    region = padded[y:y + kh, x:x + kw]
                    result[y, x] = np.sum(region * kernel)
            return result

    def gauss_(self, size: int = 5, sigma: float = 1.0) -> np.ndarray:
        center = size // 2
        x, y = np.meshgrid(np.arange(size) - center, np.arange(size) - center)
        kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        result = self.svertka_(kernel)
        return np.clip(result, 0, 255).astype(np.uint8)

    def sobel_(self) -> np.ndarray:
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        grad_x = self.svertka_(kernel_x)
        grad_y = self.svertka_(kernel_y)

        result = np.sqrt(grad_x ** 2 + grad_y ** 2)

        if result.max() > 0:
            result = (result / result.max() * 255).astype(np.uint8)
        return result


class ColorArtwork(Artwork):
    __slots__ = ('_image_type',)
    def __init__(self, image: Optional[np.ndarray] = None) -> None:
        super().__init__()
        if image is not None:
            if len(image.shape) == 2:
                self.img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                self.img = image.copy()
        self._image_type = "color"

    def halftone_(self) -> np.ndarray:
        if len(self.img.shape) == 3:
            gray = (0.299 * self.img[:, :, 2] + 0.587 * self.img[:, :, 1] + 0.114 * self.img[:, :, 0])
            return np.clip(gray, 0, 255).astype(np.uint8)
        else:
            return self.img.copy()

    def svertka_(self, kernel: Optional[np.ndarray] = None) -> np.ndarray:
        if len(self.img.shape) == 2:
            h, w = self.img.shape
            kh, kw = kernel.shape
            h_pad, w_pad = kh // 2, kw // 2
            padded = np.pad(self.img, ((h_pad, h_pad), (w_pad, w_pad)), mode='constant')
            result = np.zeros((h, w), dtype=np.float32)
            for y in range(h):
                for x in range(w):
                    region = padded[y:y + kh, x:x + kw]
                    result[y, x] = np.sum(region * kernel)
            return result
        elif len(self.img.shape) == 3:
            h, w, c = self.img.shape
            result = np.zeros((h, w, c), dtype=np.float32)
            for channel in range(c):
                single_channel = self.img[:, :, channel]
                temp = GrayscaleArtwork()
                temp.img = single_channel
                result[:, :, channel] = temp.svertka_(kernel)
            return result
    def gauss_(self, size: int = 5, sigma: float = 1.0) -> np.ndarray:
        """Фильтр Гаусса для цветного изображения"""
        center = size // 2
        x, y = np.meshgrid(np.arange(size) - center, np.arange(size) - center)
        kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        result = self.svertka_(kernel)
        return np.clip(result, 0, 255).astype(np.uint8)

    def sobel_(self) -> np.ndarray:
        """Оператор Собеля для цветного (применяется к серой версии)"""
        if self.img is None:
            raise ValueError("Изображение не загружено")

        # Для цветного сначала преобразуем в серое
        gray_img = self.halftone_()
        temp = GrayscaleArtwork(gray_img)
        return temp.sobel_()



@dataclass
class ImageProcessor:
    artworks: List[Artwork] = field(default_factory=list)
    processed: List[Artwork] = field(default_factory=list)
    output_dir: str = "paintings"

    def __post_init__(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"ImageProcessor создан. Папка: {self.output_dir}")

    def _get_painting_ids(self, csv_path: str = 'MetObjects.csv') -> List[str]:
        painting_ids = []
        try:
            with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row.get('Classification') == 'Paintings':
                        painting_ids.append(row['Object ID'])
            print(f"Найдено {len(painting_ids)} картин")
            if not painting_ids:
                print("нет картин")
                exit()
            return painting_ids
        except Exception as e:
            print(f"{e}")
            exit()

    @time_method
    def load_metadata(self, count: int = 1, csv_path: str = 'MetObjects.csv') -> None:
        print(f"\nЗагрузка метаданных ({count})")
        painting_ids = self._get_painting_ids(csv_path)
        if not painting_ids:
            print("Нет ID")
            return

        loaded = 0
        attempts = 0
        max_attempts = count * 10

        while loaded < count and attempts < max_attempts:
            attempts += 1
            random_id = random.choice(painting_ids)
            print(f"  Попытка {attempts}: пробуем ID={random_id}")

            try:
                url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{random_id}"
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()
                if not data.get('primaryImage'):
                    print("    У этого объекта нет фото")
                    continue

                # Создаём цветной объект (по умолчанию)
                artwork = ColorArtwork()
                artwork.object_id = random_id
                artwork.metadata = data
                artwork._Artwork__image_url = data.get('primaryImage')

                self.artworks.append(artwork)
                loaded += 1
                print(f"Загружена: {data.get('title', 'Unknown')}")

            except Exception as e:
                print(f" {e}")

        print(f"Загружено {loaded} картин")

    @time_method
    def load_images(self) -> None:
        print("\nЗагрузка изображений")
        for i, artwork in enumerate(self.artworks):
            print(f"  Загрузка изображения {i + 1}")

            img_url = artwork.metadata['primaryImage']
            response = requests.get(img_url, timeout=10)
            response.raise_for_status()

            filename = os.path.join(self.output_dir, f"{artwork.object_id}.jpg")
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            artwork.img = cv2.imread(filename)
            print(f"Загружено: {artwork.img.shape}")

        print("Изображения загружены!")

    @time_method
    def process_all(self, filter_type: str, **params: Any) -> List[Artwork]:
        self.processed = []
        print(f"\nПрименен метод: {filter_type}")

        for i, artwork in enumerate(self.artworks):
            print(f"  Обработка изображения {i + 1}")
            try:
                if filter_type == 'halftone_':
                    result = artwork.halftone_()
                    print("  Применен halftone_")
                elif filter_type == 'gauss':
                    size = params.get('size', 5)
                    sigma = params.get('sigma', 1.0)
                    result = artwork.gauss_(size=size, sigma=sigma)
                    print("  Применен gauss_")
                elif filter_type == 'sobel':
                    result = artwork.sobel_()
                    print("  Применен sobel_")
                else:
                    raise ValueError(f'Неизвестный фильтр: {filter_type}')

                if len(result.shape) == 2:
                    result_artwork = GrayscaleArtwork(result)
                else:
                    result_artwork = ColorArtwork(result)

                result_artwork.metadata = artwork.metadata
                result_artwork.object_id = artwork.object_id

                self.processed.append(result_artwork)

            except Exception as e:
                print(f"  {e}")

        print(f"Обработано {len(self.processed)} изображений")
        return self.processed

    @time_method
    def save_result(self, prefix: str = 'processed') -> None:
        print(f"Сохранение результатов ({prefix})")
        for i, artwork in enumerate(self.processed):
            if artwork.img is None:
                continue
            filename = f"{prefix}_{artwork.object_id or i}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            cv2.imwrite(filepath, artwork.img)
            print(f"  Сохранено {i + 1}: {filename}")

    def __str__(self) -> str:
        return (f"ImageProcessor(\n"
                f"  загружено: {len(self.artworks)},\n"
                f"  обработано: {len(self.processed)},\n"
                f"  папка: {self.output_dir}\n"
                f")")
def main():
    proc = ImageProcessor()
    proc.load_metadata(count=1)

    if proc.artworks:
        proc.load_images()

        # Сохраняем результаты от OpenCV для сравнения
        for idx, artwork in enumerate(proc.artworks):
            img = artwork.img
            object_id = artwork.object_id

            # Сохраняем результаты OpenCV
            cv2_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(f"paintings/{object_id}_halftone_cv2.jpg", cv2_gray)

            size, sigma = 5, 1.0
            center = size // 2
            x, y = np.meshgrid(np.arange(size) - center, np.arange(size) - center)
            gauss_kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            gauss_kernel = gauss_kernel / gauss_kernel.sum()
            cv2_gauss = cv2.filter2D(img, -1, gauss_kernel)
            cv2.imwrite(f"paintings/{object_id}_gauss_cv2.jpg", cv2_gauss)

            kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.filter2D(gray_img, cv2.CV_32F, kernel_x)
            sobely = cv2.filter2D(gray_img, cv2.CV_32F, kernel_y)
            cv2_sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
            cv2_sobel = (cv2_sobel / cv2_sobel.max() * 255).astype(np.uint8)
            cv2.imwrite(f"paintings/{object_id}_sobel_cv2.jpg", cv2_sobel)

            canny_result = cv2.Canny(cv2_gray, 50, 150)
            cv2.imwrite(f"paintings/{object_id}_canny.jpg", canny_result)

            gray_float = np.float32(cv2_gray)
            harris = cv2.cornerHarris(gray_float, 2, 3, 0.04)
            harris = cv2.dilate(harris, None)
            img_harris = img.copy()
            img_harris[harris > 0.01 * harris.max()] = [0, 0, 255]
            corners_count = np.sum(harris > 0.01 * harris.max())
            print(f"Для ID={object_id} найдено углов: {corners_count}")
            cv2.imwrite(f"paintings/{object_id}_harris.jpg", img_harris)

        proc.process_all('halftone_')
        proc.save_result(prefix='halftone_')

        proc.process_all('gauss', size=5, sigma=1.0)
        proc.save_result(prefix='gauss')

        proc.process_all('sobel')
        proc.save_result(prefix='sobel')

        print("\n Сохраненные файлы:")
        for file in os.listdir('paintings'):
            if file.endswith('.jpg'):
                print(f"  - {file}")
    else:
        print("Не удалось загрузить картины")


if __name__ == "__main__":
    main()