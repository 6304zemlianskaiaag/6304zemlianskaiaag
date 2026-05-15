import cv2
import csv
import os
import random
import numpy as np
import time
from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass, field
import functools
from abc import ABC, abstractmethod
import json

import aiohttp
import aiofiles
import asyncio
import argparse
from concurrent.futures import ProcessPoolExecutor


def time_method_async(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        end = time.perf_counter()
        print(f"[АСИНХРОН] {func.__name__} (PID: {os.getpid()}): {end - start:.2f} сек.")
        return result

    return wrapper


def time_method(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"[ПАРАЛЛЕЛЬ] {func.__name__} (PID: {os.getpid()}): {end - start:.2f} сек.")
        return result

    return wrapper


class Artwork(ABC):
    __slots__ = ('__img', '__metadata', '__kernel', '__object_id', '__image_url', '__path', '__index')

    def __init__(self) -> None:
        self.__img = None
        self.__metadata = None
        self.__kernel = None
        self.__object_id = None
        self.__image_url = None
        self.__path = ""
        self.__index = 0

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

    @property
    def image_url(self) -> Optional[str]:
        return self.__image_url

    @image_url.setter
    def image_url(self, url: str) -> None:
        self.__image_url = url

    @property
    def path(self) -> str:
        return self.__path

    @path.setter
    def path(self, value: str) -> None:
        self.__path = value

    @property
    def index(self) -> int:
        return self.__index

    @index.setter
    def index(self, value: int) -> None:
        self.__index = value

    def _get_task_info(self) -> str:
        return f"{self.index}_{self.object_id}"

    async def save_async(self, out_path: str) -> None:
        success, buffer = cv2.imencode(".jpg", self.img)
        if not success:
            raise ValueError(f"Ошибка кодирования: {out_path}")
        async with aiofiles.open(out_path, mode='wb') as f:
            await f.write(buffer.tobytes())

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

    @staticmethod
    def gauss_o(size: int = 5, sigma: float = 1.0) -> np.ndarray:
        center = size // 2
        x, y = np.meshgrid(np.arange(size) - center, np.arange(size) - center)
        kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel


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
        print(f"[LOG] ({self._get_task_info()}): halftone_ серый (PID: {os.getpid()})")
        return self.img.copy()

    def svertka_(self, kernel: Optional[np.ndarray] = None) -> np.ndarray:
        if len(self.img.shape) == 2:
            result = Artwork.sv_(self.img, kernel)
            return result
        else:
            raise ValueError("GrayscaleArtwork должен быть")

    def gauss_(self, size: int = 5, sigma: float = 1.0) -> np.ndarray:
        print(f"[LOG] ({self._get_task_info()}): gauss_ серый (PID: {os.getpid()})")
        kernel = Artwork.gauss_o(size, sigma)
        result = self.svertka_(kernel)
        return np.clip(result, 0, 255).astype(np.uint8)

    def sobel_(self) -> np.ndarray:
        print(f"[LOG] ({self._get_task_info()}): sobel_ серый (PID: {os.getpid()})")
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
        print(f"[LOG] ({self._get_task_info()}): halftone_ цветной (PID: {os.getpid()})")
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

    def gauss_(self, size: int = 5, sigma: float = 1.0) -> np.ndarray:
        print(f"[LOG] ({self._get_task_info()}): gauss_ цветной (PID: {os.getpid()})")
        kernel = Artwork.gauss_o(size, sigma)
        result = self.svertka_(kernel)
        return np.clip(result, 0, 255).astype(np.uint8)

    def sobel_(self) -> np.ndarray:
        print(f"[LOG] ({self._get_task_info()}): sobel_ цветной (PID: {os.getpid()})")
        h, w, c = self.img.shape
        result = np.zeros((h, w, c), dtype=np.uint8)
        for channel in range(c):
            result[:, :, channel] = Artwork.sobel_o(self.img[:, :, channel])
        return result


@dataclass
class ImageProcessor:
    artworks: List[Artwork] = field(default_factory=list)
    processed: List[Artwork] = field(default_factory=list)
    output_dir: str = "paintings"
    csv_path: str = "MetObjects.csv"  # CSV в той же папке
    api_url: str = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{}"

    def _get_painting_ids(self) -> List[str]:
        painting_ids = []
        try:
            with open(self.csv_path, mode='r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row.get('Classification') == 'Paintings':
                        painting_ids.append(row['Object ID'])
            print(f"[INFO] Найдено {len(painting_ids)} картин в CSV")
            return painting_ids
        except Exception as e:
            print(f"[ERROR] Ошибка чтения CSV: {e}")
            return []

    async def get_object_json(self, session: aiohttp.ClientSession, object_id: str):
        url = self.api_url.format(object_id)
        try:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    return await response.json()
                return None
        except Exception:
            return None

    @staticmethod
    async def download_file(session: aiohttp.ClientSession, url: str, out_path: str) -> None:
        print(f"[СКАЧИВАНИЕ] Начало (PID: {os.getpid()})")
        async with session.get(url) as response:
            if response.status == 200:
                content = await response.read()
                async with aiofiles.open(out_path, mode='wb') as f:
                    await f.write(content)
        print(f"[СКАЧИВАНИЕ] Завершено (PID: {os.getpid()})")

    @time_method_async
    async def load_metadata_and_image_async(self, session: aiohttp.ClientSession, all_ids: List[str], index: int):
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"[ЗАГРУЗКА] (№{index}): Поиск картины")

        chosen_id = None
        chosen_data = None

        for object_id in all_ids[:100]:
            data = await self.get_object_json(session, object_id)
            if data and (data.get("primaryImage") or "").strip():
                chosen_id = object_id
                chosen_data = data
                break

        if not chosen_id or not chosen_data:
            print(f"[ERROR] (№{index}): Не найдена картина с изображением")
            return None

        print(f"[ЗАГРУЗКА] (№{index}): Найдена ID={chosen_id}")

        obj_dir = os.path.join(self.output_dir, f"{index}_{chosen_id}")
        os.makedirs(obj_dir, exist_ok=True)

        img_path = os.path.join(obj_dir, f"{index}_{chosen_id}_original.jpg")
        metadata_path = os.path.join(obj_dir, "metadata.json")

        async with aiofiles.open(metadata_path, mode='w', encoding='utf-8') as f:
            await f.write(json.dumps(chosen_data, ensure_ascii=False, indent=2))

        await self.download_file(session, chosen_data["primaryImage"], img_path)

        async with aiofiles.open(img_path, mode='rb') as f:
            raw_data = await f.read()
        buffer = np.frombuffer(raw_data, np.uint8)
        img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

        artwork = ColorArtwork(img)
        artwork.object_id = chosen_id
        artwork.metadata = chosen_data
        artwork.image_url = chosen_data["primaryImage"]
        artwork.path = img_path
        artwork.index = index

        self.artworks.append(artwork)
        print(f"[ГОТОВО] (№{index}): Загружено {img.shape}")
        return artwork

    async def save_artwork(self, artwork: Artwork, suffix: str) -> None:
        dirname = os.path.dirname(artwork.path)
        filename = f"{artwork.index}_{artwork.object_id}_{suffix}.jpg"
        final_path = os.path.join(dirname, filename)
        await artwork.save_async(final_path)
        print(f"[СОХРАНЕНИЕ] ({artwork._get_task_info()}): {filename}")


    @time_method_async
    async def process_artwork_parallel(self, artwork: Artwork, executor: ProcessPoolExecutor) -> None:
        """
        ДЛЯ ОДНОГО ИЗОБРАЖЕНИЯ:
        Запускаем несколько фильтров ПАРАЛЛЕЛЬНО в разных процессах
        """
        if artwork.img is None:
            raise ValueError("Изображение не загружено")

        loop = asyncio.get_event_loop()
        print(f"\n[ПАРАЛЛЕЛЬ] ({artwork._get_task_info()}): Запуск {3} фильтров параллельно")


        tasks = [

            loop.run_in_executor(executor, artwork.halftone_),


            loop.run_in_executor(executor, artwork.gauss_),


            loop.run_in_executor(executor, artwork.sobel_),
        ]

        print(f"[ЗАПУСК] ({artwork._get_task_info()}): 3 фильтра стартуют одновременно")


        results = await asyncio.gather(*tasks)


        gray_img, gauss_img, sobel_img = results

        print(f"[ВЫПОЛНЕНО] ({artwork._get_task_info()}): Все 3 фильтра завершены")


        gray_art = GrayscaleArtwork(gray_img)
        gray_art.object_id = artwork.object_id
        gray_art.path = artwork.path
        gray_art.index = artwork.index
        await self.save_artwork(gray_art, "halftone")
        self.processed.append(gray_art)

        gauss_art = ColorArtwork(gauss_img)
        gauss_art.object_id = artwork.object_id
        gauss_art.path = artwork.path
        gauss_art.index = artwork.index
        await self.save_artwork(gauss_art, "gauss")
        self.processed.append(gauss_art)

        sobel_art = ColorArtwork(sobel_img)
        sobel_art.object_id = artwork.object_id
        sobel_art.path = artwork.path
        sobel_art.index = artwork.index
        await self.save_artwork(sobel_art, "sobel")
        self.processed.append(sobel_art)

        print(f"[ГОТОВО] ({artwork._get_task_info()}): Обработка завершена\n")

    @time_method_async
    async def run_pipeline(self, num_images: int) -> None:

        all_ids = self._get_painting_ids()
        if not all_ids:
            print(f"[ERROR] Файл не найден или нет картин: {self.csv_path}")
            return

        random.shuffle(all_ids)
        print(f"\n[INFO] Запуск обработки {num_images} изображений")
        print(f"[INFO] Каждое изображение будет обработано 3 фильтрами параллельно")
        print(f"[INFO] Итого процессов: {num_images} картин × 3 фильтра = {num_images * 3} параллельных процессов\n")

        async with aiohttp.ClientSession() as session:
            with ProcessPoolExecutor(max_workers=num_images * 3) as executor:

                async def workflow(idx: int):
                    """Обработка ОДНОЙ картины (внутри неё фильтры параллельны)"""
                    start_index = (idx - 1) * 50
                    sub_list = all_ids[start_index:start_index + 100]
                    artwork = await self.load_metadata_and_image_async(session, sub_list, idx)
                    if artwork:
                        await self.process_artwork_parallel(artwork, executor)

                print(f"[ЗАПУСК] Стартуем загрузку {num_images} картин одновременно\n")
                tasks = [workflow(i + 1) for i in range(num_images)]
                await asyncio.gather(*tasks)

        print(f"\n[ИТОГ] Загружено: {len(self.artworks)}, обработано: {len(self.processed)}")
        print(f"[ИТОГ] Всего выполнено операций: {len(self.processed)} (фильтров)")


def save_comparison(artwork: Artwork, output_dir: str = "paintings"):
    obj_id = artwork.object_id
    cv_gauss = cv2.GaussianBlur(artwork.img, (5, 5), 1.0)
    cv2.imwrite(f"{output_dir}/gauss_cv_{obj_id}.jpg", cv_gauss)

    if len(artwork.img.shape) == 3:
        h, w, c = artwork.img.shape
        cv_sobel = np.zeros((h, w, c), dtype=np.uint8)
        for channel in range(c):
            grad_x = cv2.Sobel(artwork.img[:, :, channel], cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(artwork.img[:, :, channel], cv2.CV_64F, 0, 1, ksize=3)
            channel_result = np.sqrt(grad_x ** 2 + grad_y ** 2)
            cv_sobel[:, :, channel] = np.clip(channel_result, 0, 255).astype(np.uint8)
    else:
        grad_x = cv2.Sobel(artwork.img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(artwork.img, cv2.CV_64F, 0, 1, ksize=3)
        cv_sobel = np.sqrt(grad_x ** 2 + grad_y ** 2)
        cv_sobel = np.clip(cv_sobel, 0, 255).astype(np.uint8)
    cv2.imwrite(f"{output_dir}/sobel_cv_{obj_id}.jpg", cv_sobel)
    print(f"Сравнение сохранено для {obj_id}")


async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int, nargs="?", default=2, help="Количество изображений")
    args = parser.parse_args()

    processor = ImageProcessor()
    print(f"\n{'=' * 60}")
    print(f"ЗАПУСК ЛАБОРАТОРНОЙ РАБОТЫ №4")
    print(f"{'=' * 60}")
    print(f"Количество изображений: {args.n}")
    print(f"Фильтров на изображение: 3 (halftone, gauss, sobel)")
    print(f"Всего параллельных процессов: {args.n} × 3 = {args.n * 3}")
    print(f"{'=' * 60}\n")

    start = time.perf_counter()
    await processor.run_pipeline(args.n)
    end = time.perf_counter()

    print(f"\n{'=' * 60}")
    print(f"ОБЩЕЕ ВРЕМЯ ВЫПОЛНЕНИЯ: {end - start:.2f} сек.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main_async())