import json
import os
from symspellpy import SymSpell, Verbosity
from ultralytics import YOLO
import cv2

# Инициализация SymSpell и YOLO
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
sym_spell.load_dictionary("ru-100k.txt", term_index=0, count_index=1)
model = YOLO("best.pt")  # Модель для детекции заданий


def correct_text(text):
    """Исправление орфографии"""
    suggestions = sym_spell.lookup(text, Verbosity.CLOSEST, max_edit_distance=2)
    return suggestions[0].term if suggestions else text


def extract_text_from_webres(webres_path):
    """Извлечение текста и его bounding boxes из .webRes"""
    with open(webres_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    text_boxes = []

    def parse_element(element):
        if isinstance(element, dict):
            if 'languages' in element:
                for lang in element['languages']:
                    if lang.get('lang') == 'rus':
                        for text_item in lang.get('texts', []):
                            text = text_item.get('text', '').strip()
                            if text:
                                corrected = correct_text(text)
                                box = {
                                    'x': element.get('x', 0),
                                    'y': element.get('y', 0),
                                    'w': element.get('w', 0),
                                    'h': element.get('h', 0),
                                    'text': corrected
                                }
                                text_boxes.append(box)
            for value in element.values():
                parse_element(value)
        elif isinstance(element, list):
            for item in element:
                parse_element(item)

    parse_element(data)

    return text_boxes


def get_yolo_boxes(image_path):
    results = model(image_path)
    yolo_boxes = []
    print(f"\nДетекция для {image_path}:")  # Отладочный вывод

    for result in results:
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            cls_id = int(box.cls)
            conf = box.conf.item()
            print(
                f"  Класс: {cls_id}, Метка: {model.names[cls_id]}, Conf: {conf:.2f}, BBox: {xyxy}")  # Что детектит YOLO?

            yolo_boxes.append({
                'x1': xyxy[0],
                'y1': xyxy[1],
                'x2': xyxy[2],
                'y2': xyxy[3],
                'label': cls_id  # Используем исходный класс
            })
    return yolo_boxes


def match_text_to_tasks(webres_boxes, yolo_boxes):
    task_texts = {}
    print("\nСопоставление боксов:")  # Отладочный вывод

    for yolo_box in yolo_boxes:
        task_num = yolo_box['label']
        matched_texts = []

        for webres_box in webres_boxes:
            # Проверяем, находится ли центр текста внутри YOLO-бокса
            text_center_x = webres_box['x'] + webres_box['w'] / 2
            text_center_y = webres_box['y'] + webres_box['h'] / 2

            if (yolo_box['x1'] <= text_center_x <= yolo_box['x2'] and
                    yolo_box['y1'] <= text_center_y <= yolo_box['y2']):
                matched_texts.append(webres_box['text'])
                print(f"  Найдено совпадение: Task_{task_num} -> '{webres_box['text']}'")

        if matched_texts:
            task_texts[f"Task_{task_num}"] = " ".join(matched_texts)

    return task_texts


def process_pair(webres_path, image_path):
    """Обработка пары файлов (.webRes + изображение)"""
    webres_boxes = extract_text_from_webres(webres_path)
    yolo_boxes = get_yolo_boxes(image_path)
    return match_text_to_tasks(webres_boxes, yolo_boxes)


def find_pairs(webres_dir, images_dir):
    """Поиск пар .webRes и изображений"""
    pairs = []
    webres_files = [f for f in os.listdir(webres_dir) if f.endswith('.webRes')]

    for webres_file in webres_files:
        base_name = webres_file.split('__')[0]  # Например, '2220085848_02'
        image_file = f"{base_name}.png"  # Или .jpg

        if os.path.exists(os.path.join(images_dir, image_file)):
            pairs.append((
                os.path.join(webres_dir, webres_file),
                os.path.join(images_dir, image_file)
            ))

    return pairs


def main(webres_dir, images_dir, output_file="results.txt"):
    print('start')
    pairs = find_pairs(webres_dir, images_dir)
    all_results = {}

    for webres_path, image_path in pairs:
        print(f"Обработка: {webres_path} + {image_path}")
        try:
            task_texts = process_pair(webres_path, image_path)
            all_results.update(task_texts)
        except Exception as e:
            print(f"Ошибка: {e}")

    # Сохранение результатов
    with open(output_file, 'w', encoding='utf-8') as f:
        for task, text in sorted(all_results.items()):
            f.write(f'{task}: "{text}"\n')
            print(f'{task}: "{text}"')


if __name__ == "__main__":
    webres_dir = "school"
    images_dir = "dataset/val/images"
    main(webres_dir, images_dir)