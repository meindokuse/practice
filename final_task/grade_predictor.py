import pandas as pd
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import randint

def preprocess_text(texts):
    """Предобработка текста перед векторизацией"""
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    return [str(text).lower().replace('\n', ' ').strip() for text in texts]

def train_model(X, y, save_path='grade_predictor.pkl'):
    """Обучение и сохранение модели с подбором гиперпараметров"""
    try:
        # Вычисление весов классов
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weights = dict(zip(classes, weights))
        
        # Пайплайн
        pipeline = Pipeline([
            ('preprocess', FunctionTransformer(preprocess_text)),
            ('tfidf', TfidfVectorizer(
                max_features=15000,
                ngram_range=(1, 3),
                stop_words=[],
                min_df=2,
                max_df=0.85,
                sublinear_tf=True)),
            ('clf', RandomForestClassifier(
                class_weight=class_weights,
                random_state=42,
                n_jobs=-1))
        ])
        
        # Параметры для RandomizedSearch
        param_dist = {
            'tfidf__max_features': [10000, 15000, 20000],
            'clf__n_estimators': [300, 400, 500],
            'clf__max_depth': [None, 15, 20, 25],
            'clf__min_samples_split': [2, 5, 10],
            'clf__min_samples_leaf': [1, 2, 4],
            'clf__max_features': ['sqrt', 'log2']
        }
        
        # Поиск лучших параметров
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=20,
            cv=3,
            scoring='f1_weighted',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        print("Начало подбора гиперпараметров для RandomForest...")
        search.fit(X, y)
        
        # Лучшая модель
        model = search.best_estimator_
        print("\nЛучшие параметры:", search.best_params_)
        
        # Оценка качества
        y_pred = model.predict(X)
        print("\nClassification Report на всех данных:")
        print(classification_report(y, y_pred))
        
        # Сохранение модели
        joblib.dump(model, save_path)
        print(f"\nМодель сохранена в {save_path}")
        return model
        
    except Exception as e:
        print(f"Ошибка при обучении модели: {str(e)}")
        raise

def load_model(model_path='grade_predictor.pkl'):
    """Загрузка обученной модели"""
    try:
        return joblib.load(model_path)
    except Exception as e:
        print(f"Ошибка при загрузке модели: {str(e)}")
        raise

def predict_grade(text, model=None):
    """Предсказание оценки для текста задания"""
    try:
        if model is None:
            model = load_model()
        return model.predict([text])[0]
    except Exception as e:
        print(f"Ошибка при предсказании: {str(e)}")
        raise