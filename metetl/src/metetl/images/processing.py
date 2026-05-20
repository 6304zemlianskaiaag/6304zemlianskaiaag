import cv2
import os
import numpy as np
import json
import aiohttp
import aiofiles
import asyncio
from concurrent.futures import ProcessPoolExecutor
from typing import List
from dataclasses import dataclass, field

from .models import Artwork, ColorArtwork, GrayscaleArtwork
from ..decorators import time_method_async
from ..logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ImageProcessor:
    artworks: List[Artwork] = field(default_factory=list)
    processed: List[Artwork] = field(default_factory=list)
    output_dir: str = "paintings"

    @staticmethod
    async def download_file(session: aiohttp.ClientSession, url: str, out_path: str) -> None:
        logger.debug(f"СКАЧИВАНИЕ Начало")
        async with session.get(url) as response:
            if response.status == 200:
                content = await response.read()
                async with aiofiles.open(out_path, mode='wb') as f:
                    await f.write(content)
        logger.debug(f"СКАЧИВАНИЕ Завершено")

    async def save_artwork(self, artwork: Artwork, suffix: str) -> None:
        dirname = os.path.dirname(artwork.path)
        filename = f"{artwork.index}_{artwork.object_id}_{suffix}.jpg"
        final_path = os.path.join(dirname, filename)
        await artwork.save_async(final_path)
        logger.info(f"({artwork._get_task_info()}): {filename}")

    @time_method_async
    async def process_artwork_parallel(self, artwork: Artwork, executor: ProcessPoolExecutor) -> None:
        if artwork.img is None:
            raise ValueError("Изображение не загружено")

        loop = asyncio.get_event_loop()
        logger.info(f"({artwork._get_task_info()}): Запуск 3 фильтров")

        tasks = [
            loop.run_in_executor(executor, artwork.halftone_),
            loop.run_in_executor(executor, artwork.gauss_),
            loop.run_in_executor(executor, artwork.sobel_),
        ]

        results = await asyncio.gather(*tasks)
        gray_img, gauss_img, sobel_img = results

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

        logger.info(f"({artwork._get_task_info()}): Обработка завершена")


def process_(input_json: str, output_dir: str, num: int = 5):
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)  # словарь

 #   if isinstance(data, dict):
    #    data = [data]
    selected = data[:num]

    logger.info(f"Обработка {len(selected)} изображений из JSON")

    async def run():
        processor = ImageProcessor(output_dir=output_dir)

        async with aiohttp.ClientSession() as session:
            with ProcessPoolExecutor(max_workers=num * 3) as executor:
                for idx, item in enumerate(selected, 1):
                    object_id = item.get('object_id')
                    image_url = item.get('url')
                    title = item.get('title', 'Unknown')

                    logger.info(f"[{idx}] {title} (ID: {object_id})")

                    obj_dir = os.path.join(output_dir, f"{idx}_{object_id}")
                    os.makedirs(obj_dir, exist_ok=True)
                    img_path = os.path.join(obj_dir, f"{idx}_{object_id}_original.jpg")

                    await processor.download_file(session, image_url, img_path)

                    async with aiofiles.open(img_path, 'rb') as f:
                        raw = await f.read()
                    buffer = np.frombuffer(raw, np.uint8)
                    img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

                    artwork = ColorArtwork(img)
                    artwork.object_id = object_id
                    artwork.path = img_path
                    artwork.index = idx

                    await processor.process_artwork_parallel(artwork, executor)
                    processor.artworks.append(artwork)

        return len(processor.artworks)

    return asyncio.run(run())
