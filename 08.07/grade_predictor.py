import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def preprocess_text(texts):
    """Предобработка текста перед векторизацией"""
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    return [str(text).lower().replace('\n', ' ').strip() for text in texts]

def train_model(X, y, save_path='grade_predictor.pkl'):
    """Обучение и сохранение модели"""
    model = Pipeline([
        ('preprocess', FunctionTransformer(preprocess_text)),
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words=['который', 'которые', 'это'])),
        ('clf', RandomForestClassifier(
            n_estimators=150,
            class_weight='balanced',
            random_state=42))
    ])
    
    # Разделение на train/test для оценки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train)
    
    # Оценка качества
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Сохранение модели
    joblib.dump(model, save_path)
    print(f"Модель сохранена в {save_path}")
    return model

def load_model(model_path='grade_predictor.pkl'):
    """Загрузка обученной модели"""
    return joblib.load(model_path)

def predict_grade(text, model=None):
    """Предсказание оценки для текста задания"""
    if model is None:
        model = load_model()
    return model.predict([text])[0]