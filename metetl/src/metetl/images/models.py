import cv2
import csv
import os
import random
import requests
import numpy as np
import time
import aiofiles
from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass, field
import functools
from abc import ABC, abstractmethod
from ..logging_config import get_logger
from ..decorators import time_method, time_method_async
logger = get_logger(__name__)


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
        logger.debug(f"({self._get_task_info()}): halftone_ серый (PID: {os.getpid()})")
        return self.img.copy()

    def svertka_(self, kernel: Optional[np.ndarray] = None) -> np.ndarray:
        if len(self.img.shape) == 2:
            result = Artwork.sv_(self.img, kernel)
            return result
        else:
            raise ValueError("GrayscaleArtwork должен быть")

    def gauss_(self, size: int = 5, sigma: float = 1.0) -> np.ndarray:
        logger.debug(f"({self._get_task_info()}): gauss_ серый (PID: {os.getpid()})")
        kernel = Artwork.gauss_o(size, sigma)
        result = self.svertka_(kernel)
        return np.clip(result, 0, 255).astype(np.uint8)

    def sobel_(self) -> np.ndarray:
        logger.debug(f"({self._get_task_info()}): sobel_ серый (PID: {os.getpid()})")
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
        logger.debug(f"({self._get_task_info()}): halftone_ цветной (PID: {os.getpid()})")
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
        logger.debug(f"({self._get_task_info()}): gauss_ цветной (PID: {os.getpid()})")
        kernel = Artwork.gauss_o(size, sigma)
        result = self.svertka_(kernel)
        return np.clip(result, 0, 255).astype(np.uint8)

    def sobel_(self) -> np.ndarray:
        logger.debug(f"({self._get_task_info()}): sobel_ цветной (PID: {os.getpid()})")
        h, w, c = self.img.shape
        result = np.zeros((h, w, c), dtype=np.uint8)
        for channel in range(c):
            result[:, :, channel] = Artwork.sobel_o(self.img[:, :, channel])
        return result
