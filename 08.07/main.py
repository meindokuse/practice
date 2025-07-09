import json
import os
import pandas as pd
from symspellpy import SymSpell, Verbosity
from ultralytics import YOLO
from grade_predictor import train_model, predict_grade, load_model
import joblib

# Инициализация компонентов
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
sym_spell.load_dictionary("ru-100k.txt", term_index=0, count_index=1)
model_yolo = YOLO("best.pt")

def correct_text(text):
    """Исправление орфографии с SymSpell"""
    suggestions = sym_spell.lookup(text, Verbosity.CLOSEST, max_edit_distance=2)
    return suggestions[0].term if suggestions else text

def extract_text_from_webres(webres_path):
    """Извлечение текста из .webRes файла"""
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
    """Получение bounding boxes заданий с YOLO"""
    results = model_yolo(image_path)
    yolo_boxes = []
    
    for result in results:
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            yolo_boxes.append({
                'x1': xyxy[0],
                'y1': xyxy[1],
                'x2': xyxy[2],
                'y3': xyxy[3],
                'label': int(box.cls.item())
            })
    return yolo_boxes

def match_text_to_tasks(webres_boxes, yolo_boxes):
    """Сопоставление текста с заданиями"""
    task_texts = {}
    
    for yolo_box in yolo_boxes:
        task_num = yolo_box['label']
        matched_texts = []
        
        for webres_box in webres_boxes:
            text_center_x = webres_box['x'] + webres_box['w'] / 2
            text_center_y = webres_box['y'] + webres_box['h'] / 2
            
            if (yolo_box['x1'] <= text_center_x <= yolo_box['x2'] and
                yolo_box['y1'] <= text_center_y <= yolo_box['y3']):
                matched_texts.append(webres_box['text'])
        
        if matched_texts:
            task_texts[f"Task_2{task_num + 2}"] = " ".join(matched_texts)
    
    return task_texts

def load_grades(grades_path):
    """Загрузка оценок из JSON"""
    with open(grades_path) as f:
        return json.load(f)

def prepare_training_data(task_texts, grades_data):
    print("Raw data:", task_texts, grades_data)  # Отладочный вывод
    training_data = []
    
    for task, text in task_texts.items():
        print(f"Processing: {task} = {text}")  # Отладка
        try:
            task_num = int(task.split('_')[1])
            print(f"Task number: {task_num}")  # Отладка
            
            if 22 <= task_num <= 28:  # Фильтруем нужные задания
                grade_pos = task_num - 22  # Преобразуем номер в позицию в маске
                grade_mask = grades_data['mask'].split(')')[:-1]
                print(f"Grade mask positions: {grade_mask}")  # Отладка
                
                if grade_pos < len(grade_mask):
                    grade = int(grade_mask[grade_pos][0])  # Берём первую цифру
                    training_data.append({
                        'task_num': task_num,  # Сохраняем ИСХОДНЫЙ номер задания!
                        'text': text,
                        'grade': grade
                    })
                    print(f"Added: Task {task_num}, Grade {grade}")  # Отладка
                else:
                    print(f"No grade for task {task_num} in mask")
            else:
                print(f"Task {task_num} out of range (22-28)")
        except Exception as e:
            print(f"Error processing {task}: {e}")
    
    return pd.DataFrame(training_data)

def process_pair(webres_path, image_path, grades_dir):
    """Обработка пары файлов (webres + изображение)"""
    webres_boxes = extract_text_from_webres(webres_path)
    yolo_boxes = get_yolo_boxes(image_path)
    task_texts = match_text_to_tasks(webres_boxes, yolo_boxes)
    
    # Получение оценок
    base_name = os.path.basename(webres_path).split('_')[0]
    print(base_name)
    grades_path = os.path.join(grades_dir, f"{base_name}.json")
    print(grades_path)
    
    if os.path.exists(grades_path):
        grades_data = load_grades(grades_path)
        df_train = prepare_training_data(task_texts, grades_data)
        return task_texts, df_train
    
    print('путь не найден до оценок')
    
    return task_texts, None

def find_pairs(webres_dir, images_dir):
    """Поиск пар файлов для обработки"""
    pairs = []
    webres_files = [f for f in os.listdir(webres_dir) if f.endswith('.webRes')]
    
    for webres_file in webres_files:
        base_name = webres_file.split('__')[0]
        image_file = f"{base_name}.png"
        image_path = os.path.join(images_dir, image_file)
        
        if os.path.exists(image_path):
            pairs.append((
                os.path.join(webres_dir, webres_file),
                image_path
            ))
    
    return pairs

def save_results(all_results, output_file):
    """Сохранение результатов в файл"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in all_results:
            for task, text in result.items():
                f.write(f'{task}: "{text}"\n')

def main(webres_dir, images_dir, grades_dir, output_file="results.txt"):
    """Основной пайплайн обработки"""
    pairs = find_pairs(webres_dir, images_dir)
    all_results = []
    training_dfs = []
    
    print(f"Найдено {len(pairs)} пар файлов для обработки")
    
    for webres_path, image_path in pairs:
        try:
            print(f"Обработка: {os.path.basename(webres_path)}...")
            task_texts, df_train = process_pair(webres_path, image_path, grades_dir)
            
            if task_texts:
                all_results.append(task_texts)
            
            if df_train is not None and not df_train.empty:
                training_dfs.append(df_train)
                
        except Exception as e:
            print(f"Ошибка при обработке {webres_path}: {str(e)}")
    
    # Сохранение всех результатов
    save_results(all_results, output_file)
    print(f"\nРезультаты сохранены в {output_file}")
    
    # Обучение модели если есть данные
    if training_dfs:
        df_train = pd.concat(training_dfs)
        print(f"\nОбучение модели на {len(df_train)} примерах...")
        
        # Разделяем на features (X) и target (y)
        X = df_train['text']
        y = df_train['grade']
        
        # Обучение и сохранение модели
        model = train_model(X, y)
        
        # Пример предсказания
        if not df_train.empty:
            sample = df_train.iloc[0]
            predicted = predict_grade(sample['text'], model)
            print(f"\nТестовое предсказание:")
            print(f"Текст: {sample['text'][:100]}...")
            print(f"Истинная оценка: {sample['grade']}")
            print(f"Предсказанная оценка: {predicted}")
        
        # Сохранение данных обучения для анализа
        df_train.to_csv('training_data.csv', index=False)
        print("Данные для обучения сохранены в training_data.csv")
    else:
        print("\nНе найдено данных для обучения модели")

if __name__ == "__main__":
    # Получаем абсолютный путь к директории скрипта
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Поднимаемся на уровень выше
    parent_dir = os.path.dirname(script_dir)
    
    config = {
        'webres_dir': os.path.join(parent_dir, "school"),
        'images_dir': os.path.join(parent_dir, "301"),
        'grades_dir': os.path.join(parent_dir, "301"),
        'output_file': os.path.join(script_dir, "results.txt")  # results.txt сохраняем в директории скрипта
    }
    
    # Проверка существования директорий
    for dir_type, dir_path in [('webres', config['webres_dir']), 
                              ('images', config['images_dir']),
                              ('grades', config['grades_dir'])]:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Директория {dir_type} не найдена: {dir_path}")
    
    main(**config)