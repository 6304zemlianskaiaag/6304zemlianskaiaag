import csv, json, os, random, requests
from ..logging_config import get_logger
logger = get_logger(__name__)

def prepare_json(csv_path, output_json, limit=5):
    paintings = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_ids = [row['Object ID'] for row in reader if row.get('Classification') == 'Paintings']
    for _ in range(limit * 5):
        if len(paintings) >= limit:
            break
        random_id = random.choice(all_ids)
        try:
            data = requests.get(f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{random_id}",
                                timeout=10).json()
            if data.get('primaryImage'):
                paintings.append({
                    'object_id': str(data['objectID']),
                    'url': data['primaryImage'],
                    'title': data.get('title', 'Unknown')
                })
                logger.info(f"Найдено: {paintings[-1]['title'][:50]}")
        except:
            pass
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(paintings, f, indent=2)

    logger.info(f"Сохранено {len(paintings)} картин в {output_json}")
    return paintings