"""Скрипт для скачивания изображений из Met Museum API."""

import csv
import json
import os
import random

import requests


with open('MetObjects.csv', encoding='utf-8-sig') as file:
    reader = csv.DictReader(file)
    painting_object_n = []
    for row in reader:
        if row['Classification'] == 'Paintings':
            painting_object_n.append(row['Object Number'])

print(f"Всего картин в списке: {len(painting_object_n)}")

max_attempts = 30
found = False
for i in range(max_attempts):
    random_object = random.choice(painting_object_n)
    print(f"\nПопытка {i + 1}: пробуем номер {random_object}")

    # Ищем числовой ID по этому номеру
    search_url = (
        f"https://collectionapi.metmuseum.org/public/collection/v1/"
        f"search?q={random_object}&hasImages=true"
    )

    try:
        search_response = requests.get(search_url)
        search_response.raise_for_status()
        search_data = search_response.json()

        # Если нашли ID
        if search_data['total'] > 0 and search_data.get('objectIDs'):
            object_id = search_data['objectIDs'][0]
            print(f"Найден ID: {object_id}")

            # Получаем подробную информацию по ID
            detail_url = (
                f"https://collectionapi.metmuseum.org/public/collection/v1/"
                f"objects/{object_id}"
            )
            detail_response = requests.get(detail_url)
            detail_response.raise_for_status()
            data = detail_response.json()

            # Проверяем, есть ли фото
            image_url = data.get('primaryImage')
            if image_url:
                print(f"Фото: {image_url}")
                found = True
                break
            else:
                print("У этого объекта нет фото")
        else:
            print("Номер не найден в API")
    except Exception as e:
        print(f"Ошибка: {e}")

if not found:
    print(f"\nНе удалось найти картину с фото за {max_attempts} попыток")
    exit()

# Создаём папку для картинок
os.makedirs('paintings', exist_ok=True)

# Сохраняем картинку
img_filename = os.path.join('paintings', f"{object_id}.jpg")
img_response = requests.get(image_url, stream=True)
with open(img_filename, 'wb') as f:
    for chunk in img_response.iter_content(chunk_size=8192):
        f.write(chunk)

print(f"Изображение сохранено: {img_filename}")

# Сохраняем метаданные
json_filename = os.path.join('paintings', f"{object_id}.json")
with open(json_filename, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Метаданные сохранены: {json_filename}")
