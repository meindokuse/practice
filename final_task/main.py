import json
import os
import pandas as pd
import joblib
from symspellpy import SymSpell, Verbosity
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Инициализация компонентов
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
sym_spell.load_dictionary("ru-100k.txt", term_index=0, count_index=1)
model_yolo = YOLO("best_3.pt")

# Загрузка обученной модели
try:
    model = joblib.load('rf_predictor.pkl')  # Просто загружаем файл модели
    print("Модель успешно загружена")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    exit()

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
            task_texts[f"Task_{task_num + 22}"] = " ".join(matched_texts)
    
    return task_texts

def load_grades(grades_path):
    """Загрузка оценок из JSON"""
    with open(grades_path) as f:
        return json.load(f)

def prepare_prediction_data(task_texts, grades_data=None):
    """Подготовка данных для предсказания"""
    prediction_data = []
    
    for task, text in task_texts.items():
        try:
            task_num = int(task.split('_')[1])
            if 22 <= task_num <= 28:
                pred_item = {
                    'Задание': task,
                    'Текст': text,
                    'Предсказанная оценка': model.predict([text])[0]  # Используем загруженную модель
                }
                
                if grades_data:
                    grade_pos = task_num - 22
                    grade_mask = grades_data['mask'].split(')')[:-1]
                    if grade_pos < len(grade_mask):
                        pred_item['Истинная оценка'] = int(grade_mask[grade_pos][0])
                
                prediction_data.append(pred_item)
        except Exception as e:
            print(f"Ошибка обработки {task}: {e}")
    
    return pd.DataFrame(prediction_data)

def save_results(df, output_excel="results.xlsx", output_json="metrics.json"):
    """Сохранение результатов"""
    # Сохранение в Excel
    df.to_excel(output_excel, index=False)
    print(f"Результаты сохранены в {output_excel}")
    
    # Если есть истинные оценки, сохраняем метрики
    if 'Истинная оценка' in df.columns:
        y_true = df['Истинная оценка']
        y_pred = df['Предсказанная оценка']
        
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_scores": {
                str(i): f1_score(y_true, y_pred, average=None)[i] for i in range(4)
            },
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }
        
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        print(f"Метрики сохранены в {output_json}")

def find_pairs(webres_dir, images_dir):
    """Поиск пар файлов"""
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

def main():
    # Конфигурация путей
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    config = {
        'webres_dir': os.path.join(parent_dir, "school"),
        'images_dir': os.path.join(parent_dir, "301"),
        'grades_dir': os.path.join(parent_dir, "301"),
        'output_excel': os.path.join(script_dir, "results.xlsx"),
        'output_json': os.path.join(script_dir, "metrics.json")
    }
    
    # Поиск и обработка пар файлов
    pairs = find_pairs(config['webres_dir'], config['images_dir'])
    all_results = []
    
    for webres_path, image_path in pairs:
        try:
            webres_boxes = extract_text_from_webres(webres_path)
            yolo_boxes = get_yolo_boxes(image_path)
            task_texts = match_text_to_tasks(webres_boxes, yolo_boxes)
            
            base_name = os.path.basename(webres_path).split('_')[0]
            grades_path = os.path.join(config['grades_dir'], f"{base_name}.json")
            
            grades_data = None
            if os.path.exists(grades_path):
                grades_data = load_grades(grades_path)
            
            df_pred = prepare_prediction_data(task_texts, grades_data)
            all_results.append(df_pred)
            
        except Exception as e:
            print(f"Ошибка обработки {webres_path}: {str(e)}")
    
    if all_results:
        df_final = pd.concat(all_results)
        save_results(df_final, config['output_excel'], config['output_json'])
        
        # Пример вывода
        print("\nПримеры предсказаний:")
        print(df_final.head())
    else:
        print("Не найдено данных для обработки")

if __name__ == "__main__":
    main()