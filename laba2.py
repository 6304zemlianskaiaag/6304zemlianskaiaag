
import cv2
import csv
import json
import os
import random
import requests
import numpy as np
import time
from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass, field
import functools

def time_method(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f" Время выполнения {func.__name__}: {end - start:.2f} сек.")
        return result
    return wrapper

class Artwork:
    __slots__ = ('__img', '__metadata', '__kernel', '__object_id', '__image_url')

    def __init__(self) -> None:
        self.__img = None
        self.__metadata = None
        self.__kernel = None
        self.__object_id = None
        self.__image_url = None

    @property
    def img(self) -> Optional[np.ndarray]:
        return self.__img

    @img.setter
    def img(self, val: Optional[np.ndarray]) -> None:
        if val is not None and not isinstance(val, np.ndarray):
            raise TypeError(" Ошибка!!!")
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
        raise ValueError(' Ошибка!!!')

    @kernel.setter
    def kernel(self, matrix: np.ndarray) -> None:
        h, w = matrix.shape
        if h == w:
            self.__kernel = matrix
        else:
            raise ValueError(' Ошибка!!!')

    @property
    def object_id(self) -> Optional[str]:
        return self.__object_id

    @object_id.setter
    def object_id(self, value: str) -> None:
        self.__object_id = value

    def halftone_(self) -> np.ndarray:
        if len(self.__img.shape) == 3:
         gray = (0.299 * self.__img[:, :, 2] + 0.587 * self.__img[:, :, 1] + 0.114 * self.__img[:, :, 0])
         return np.clip(gray, 0, 255).astype(np.uint8)
        else:
            return self.__img.copy()

    def svertka_(self, kernel: Optional[np.ndarray] = None) -> np.ndarray:
        if len(self.__img.shape) == 2:
            h, w = self.__img.shape
            kh, kw = kernel.shape
            h_pad, w_pad = kh // 2, kw // 2
            padded = np.pad(self.__img, ((h_pad, h_pad), (w_pad, w_pad)), mode='constant')
            result = np.zeros((h, w), dtype=np.float32)
            for y in range(h):
                for x in range(w):
                    region = padded[y:y + kh, x:x + kw]
                    result[y, x] = np.sum(region * kernel)
            return np.clip(result, 0, 255).astype(np.uint8)
        elif len(self.__img.shape) == 3:
            h, w, c = self.__img.shape
            result = np.zeros((h, w, c), dtype=np.float32)
            for channel in range(c):
                single_channel = self.__img[:, :, channel]
                temp = Artwork()
                temp.__img = single_channel
                result[:, :, channel] = temp.svertka_(kernel)
            return np.clip(result, 0, 255).astype(np.uint8)

    def gauss_(self, size: int = 5, sigma: float = 1.0) -> np.ndarray:
        center = size // 2
        x, y = np.meshgrid(np.arange(size) - center, np.arange(size) - center)
        kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return self.svertka_(kernel)

    def sobel_(self) -> np.ndarray:
        if self.__img is None:
            raise ValueError("Изображение не загружено")
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        if len(self.__img.shape) == 3:
            gray_image = self.halftone_()
        else:
            gray_image = self.__img.copy()
        temp = Artwork()
        temp.__img = gray_image
        grad_x = temp.svertka_(kernel_x)
        grad_y = temp.svertka_(kernel_y)
        result = np.sqrt(grad_x ** 2 + grad_y ** 2)
        if result.max() > 0:
            result = (result / result.max() * 255).astype(np.uint8)
        return result

    def __add__(self, other: Union['Artwork', int, float]) -> 'Artwork':
        new_object = Artwork()
        if isinstance(other, (int, float)):
            result = np.clip(self.img.astype(np.int16) + other, 0, 255).astype(np.uint8)
            new_object.img = result
            return new_object
        elif isinstance(other, Artwork):
            if self.img.shape != other.img.shape:
                raise ValueError(f"Размеры изображений не совпадают!")
            result = np.clip((self.img.astype(np.int16) + other.img.astype(np.int16)) // 2, 0, 255).astype(np.uint8)
            new_object.img = result
            return new_object

    def __radd__(self, other: Union[int, float]) -> 'Artwork':
        return self.__add__(other)

    def __str__(self) -> str:
        return f"Изображение (ID: {self.object_id}, размер:{self.img.shape})"
@dataclass
class ImageProcessor:
    artworks: List[Artwork] = field(default_factory=list)
    processed: List[Artwork] = field(default_factory=list)
    output_dir: str = "paintings"

    def __post_init__(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        print(f" ImageProcessor создан. Папка: {self.output_dir}")

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
                print(" нет картин ")
                exit()
            return painting_ids
        except Exception as e:
            print(f"{e}")
            exit()

    @time_method
    def load_metadata_batch(self, count: int = 1, csv_path: str = 'MetObjects.csv') -> None:
        print(f"\nЗагрузка метаданных ({count} )")
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

                artwork = Artwork()
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
        print("\n Загрузка изображений")
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

        print(" Изображения загружены!")

    def save_metadata(self, save_dir: str = 'paintings') -> None:
        print("\n Сохранение метаданных")
        os.makedirs(save_dir, exist_ok=True)
        for artwork in self.artworks:
            filename = os.path.join(save_dir, f"{artwork.object_id}.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(artwork.metadata, f, indent=2, ensure_ascii=False)
            print(f" {filename}")
        print(" Метаданные сохранены!")

    @time_method
    def process_all(self, filter_type: str, **params: Any) -> List[Artwork]:
        self.processed = []
        print(f"\n Применен метод: {filter_type}")
        for i, artwork in enumerate(self.artworks):
            print(f"  Обработка изображения {i + 1}")
            try:
                if filter_type == 'gray':
                    result = artwork.halftone_()
                    print(" Применен halftone_ ")

                elif filter_type == 'gauss':
                    size = params.get('size', 5)
                    sigma = params.get('sigma', 1.0)
                    result = artwork.gauss_(size=size, sigma=sigma)
                    print("Применен gauss_")

                elif filter_type == 'sobel':
                    result = artwork.sobel_()
                    print("    Применен sobel_")
                else:
                    raise ValueError(f'Неизвестный фильтр: {filter_type}')

                print(" Обработка  применена к изображению!")

                result_artwork = Artwork()
                result_artwork.img = result
                result_artwork.metadata = artwork.metadata
                result_artwork.object_id = artwork.object_id

                self.processed.append(result_artwork)

            except Exception as e:
                print(f"{e}")

        print(f" Обработано {len(self.processed)} изображений")
        return self.processed

    @time_method
    def save_result(self, prefix: str = 'processed') -> None:
        print(f" Сохранение результатов ( {prefix}")
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
    proc.load_metadata_batch(count=1)

    if proc.artworks:
        proc.load_images()

        proc.process_all('gray')
        proc.save_result(prefix='gray')


        proc.process_all('gauss', size=5, sigma=1.0)
        proc.save_result(prefix='gauss')


        proc.process_all('sobel')
        proc.save_result(prefix='sobel')


        print(" Сохраненные файлы:")
        for file in os.listdir('paintings'):
            if file.endswith('.jpg'):
                print(f"  - {file}")

    else:
        print(" Не удалось загрузить картины")
    art1 = Artwork()
    art1.img = cv2.imread("paintings/73370.jpg")

    art2 = Artwork()
    art2.img = cv2.imread("paintings/sobel_73370.jpg")
    # Смешиваем (усредняем)
    mixed = art1 + art2
    # Сохраняем
    cv2.imwrite("paintings/mixed.jpg", mixed.img)

if __name__ == "__main__":
    main()