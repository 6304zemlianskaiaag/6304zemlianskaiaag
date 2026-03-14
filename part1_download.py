"""Скрипт для скачивания изображений из Met Museum API."""

import csv
import json
import os
import random
import requests

id_paint = []
with open('MetObjects.csv', mode='r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        if row.get('Classification') == 'Paintings':
            id_paint.append(row['Object ID'])

print(f"Всего картин в списке: {len(id_paint)}")

max_ = 30
found = False
for i in range(max_):
    random_id = random.choice(id_paint)
    print(f"Попытка {i + 1}: пробуем номер {random_id}")
    url_ = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{random_id}"
    try:
        detail_ = requests.get(url_)
        detail_.raise_for_status()
        data = detail_.json()
        image_url = data.get('primaryImage')

        if image_url:
            print(f"Нашли фото: {image_url}")
            found = True
            break
        else:
            print("У этого объекта нет фото")
    except Exception as e:
        print(f"Ошибка: {e}")

if not found:
    print(f"Не удалось найти картину за {max_} попыток")
    exit()

os.makedirs('paintings', exist_ok=True)
img_filename = os.path.join('paintings', f"{random_id}.jpg")
img_ = requests.get(image_url, stream=True)

with open(img_filename, 'wb') as f:  # 'wb' = write binary
    for chunk in img_.iter_content(chunk_size=8192):
        f.write(chunk)
print(f"Изображение сохранено: {img_filename}")


json_filename = os.path.join('paintings', f"{random_id}.json")
with open(json_filename, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Метаданные сохранены: {json_filename}")