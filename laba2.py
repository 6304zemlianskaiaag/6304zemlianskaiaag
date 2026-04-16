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
            if type(self) != type(other):
                raise TypeError(f"Нельзя складывать {type(self).__name__} с {type(other).__name__}")
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

    @staticmethod
    def sv_(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        kh, kw = kernel.shape
        h, w = img.shape
        pad_h, pad_w = kh // 2, kw // 2

        padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        result = np.zeros((h, w), dtype=np.float32)

        for y in range(h):
            for x in range(w):
                region = padded[y:y + kh, x:x + kw]
                result[y, x] = np.sum(region * kernel)

        return result

    @staticmethod
    def sobel_o(img: np.ndarray) -> np.ndarray:
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

        grad_x = Artwork.sv_(img, kernel_x)
        grad_y = Artwork.sv_(img, kernel_y)
        result = np.sqrt(grad_x.astype(np.float32) ** 2 + grad_y.astype(np.float32) ** 2)

        if result.max() > 0:
            result = (result / result.max() * 255).astype(np.uint8)

        return result

    def gauss_(self, size: int = 5, sigma: float = 1.0) -> np.ndarray:
        center = size // 2
        x, y = np.meshgrid(np.arange(size) - center, np.arange(size) - center)
        kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        result = self.svertka_(kernel)  # ← полиморфизм
        return np.clip(result, 0, 255).astype(np.uint8)




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
            result = Artwork.sv_(self.img, kernel)
            return result
        else:
            raise ValueError("GrayscaleArtwork должен быть")


    def sobel_(self) -> np.ndarray:
        if self.img is None:
            raise ValueError("Изображение не загружено")
        return Artwork.sobel_o(self.img)

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
        if len(self.img.shape) == 3:
            h, w, c = self.img.shape
            result = np.zeros((h, w, c), dtype=np.float32)
            for channel in range(c):
                single_channel = self.img[:, :, channel]
                result[:, :, channel] = Artwork.sv_(single_channel, kernel)
            return result
        else:
            result = Artwork.sv_(self.img, kernel)
            return result

    def sobel_(self) -> np.ndarray:
        """Цветной Собель"""
        if self.img is None:
            raise ValueError("Изображение не загружено")

        # Сохраняем ссылку на изображение, чтобы избежать проблем
        img_copy = self.img

        h, w, c = img_copy.shape
        result = np.zeros((h, w, c), dtype=np.uint8)

        for channel in range(c):
            result[:, :, channel] = Artwork.sobel_o(img_copy[:, :, channel])

        return result


@dataclass
class ImageProcessor:
    artworks: List[Artwork] = field(default_factory=list)
    processed: List[Artwork] = field(default_factory=list)
    output_dir: str = "paintings"
    def __init__(self, artworks,prosssed,output_dir):
        self.artworks=artworks



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


def save_comparison(artwork: Artwork, output_dir: str = "paintings"):
    obj_id = artwork.object_id or "unknown"

    # Гаусс
    cv_gauss = cv2.GaussianBlur(artwork.img, (5, 5), 1.0)
    cv2.imwrite(f"{output_dir}/gauss_cv_{obj_id}.jpg", cv_gauss)

    # Собель (цветной для OpenCV)
    if len(artwork.img.shape) == 3:
        # Цветное изображение — применяем Собель к каждому каналу
        h, w, c = artwork.img.shape
        cv_sobel = np.zeros((h, w, c), dtype=np.uint8)

        for channel in range(c):
            grad_x = cv2.Sobel(artwork.img[:, :, channel], cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(artwork.img[:, :, channel], cv2.CV_64F, 0, 1, ksize=3)
            channel_result = np.sqrt(grad_x ** 2 + grad_y ** 2)
            cv_sobel[:, :, channel] = np.clip(channel_result, 0, 255).astype(np.uint8)
    else:
        # Ч/б изображение
        grad_x = cv2.Sobel(artwork.img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(artwork.img, cv2.CV_64F, 0, 1, ksize=3)
        cv_sobel = np.sqrt(grad_x ** 2 + grad_y ** 2)
        cv_sobel = np.clip(cv_sobel, 0, 255).astype(np.uint8)

    cv2.imwrite(f"{output_dir}/sobel_cv_{obj_id}.jpg", cv_sobel)

    print(f"Сравнение сохранено для {obj_id}")
def main():
    proc = ImageProcessor()
    proc.load_metadata(count=1)

    if proc.artworks:
        proc.load_images()

        # Ваши обработки
        proc.process_all('halftone_')
        proc.save_result(prefix='halftone_')

        proc.process_all('gauss', size=5, sigma=1.0)
        proc.save_result(prefix='gauss')

        proc.process_all('sobel')
        proc.save_result(prefix='sobel')

        # === ВЫЗОВ СРАВНЕНИЯ ===
        print("\n=== Сравнение с OpenCV ===")
        for artwork in proc.artworks:
            save_comparison(artwork)  # <-- ВЫЗОВ ФУНКЦИИ

        print("\n Сохраненные файлы:")
        for file in os.listdir('paintings'):
            if file.endswith('.jpg'):
                print(f"  - {file}")
    else:
        print("Не удалось загрузить картины")


if __name__ == "__main__":
    main()