
# ==============================================================================
# Шаг 1: Подготовка среды и загрузка данных
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Для машинного обучения
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC # Support Vector Machine
from sklearn.metrics import accuracy_score, classification_report

# Можно попробовать XGBoost, но пока сфокусируемся на более "безопасных" вариантах
# !pip install xgboost
from xgboost import XGBClassifier

# Загрузка данных
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    test_passenger_ids = test_df['PassengerId']
except FileNotFoundError:
    print("Ошибка: Файлы train.csv и test.csv не найдены.")
    print("Пожалуйста, загрузите их в текущую сессию Google Colab через левую панель (иконка папки -> стрелочка вверх).")
    import sys
    sys.exit("Выход из программы.")

print("Данные успешно загружены.")
print(f"Размер train_df: {train_df.shape}")
print(f"Размер test_df: {test_df.shape}")

# ==============================================================================
# Шаг 2: Расширенная предобработка данных (Feature Engineering & Preprocessing)
# Упрощенная и более стабильная версия
# ==============================================================================

print("\n--- Начало расширенной предобработки данных (упрощенная версия) ---")

# Объединяем train и test для единообразной обработки
train_target = train_df['Survived']
train_df_processed = train_df.drop('Survived', axis=1)

train_df_processed['is_train'] = 1
test_df['is_train'] = 0

combined_df = pd.concat([train_df_processed, test_df], ignore_index=True)

# 1. Обработка пропусков
combined_df['Age'].fillna(combined_df['Age'].median(), inplace=True)
combined_df['Embarked'].fillna(combined_df['Embarked'].mode()[0], inplace=True)
combined_df['Fare'].fillna(combined_df['Fare'].median(), inplace=True)

# 2. Инженерия признаков (более консервативная)

# FamilySize: Размер семьи (SibSp + Parch + сам пассажир)
combined_df['FamilySize'] = combined_df['SibSp'] + combined_df['Parch'] + 1
# IsAlone: 1, если пассажир едет один (FamilySize = 1), иначе 0
combined_df['IsAlone'] = (combined_df['FamilySize'] == 1).astype(int)

# Title: Извлечение титула из имени
combined_df['Title'] = combined_df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
combined_df['Title'] = combined_df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
combined_df['Title'] = combined_df['Title'].replace('Mlle', 'Miss')
combined_df['Title'] = combined_df['Title'].replace('Ms', 'Miss')
combined_df['Title'] = combined_df['Title'].replace('Mme', 'Mrs')
# Ручное кодирование титулов (более предсказуемо, чем One-Hot для небольшого количества категорий)
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}
combined_df['Title'] = combined_df['Title'].map(title_mapping).fillna(0).astype(int) # Заполняем пропуски 0 (Mr)

# Pclass * Fare: новый признак, который может учитывать "богатство"
combined_df['Pclass_Fare'] = combined_df['Pclass'] * combined_df['Fare']

# 3. Преобразование категориальных признаков в числовой формат

# Sex: 'male' -> 0, 'female' -> 1
combined_df['Sex'] = combined_df['Sex'].map({'male': 0, 'female': 1})

# Embarked (One-Hot Encoding, так как 3 категории)
combined_df = pd.get_dummies(combined_df, columns=['Embarked'], prefix='Embarked', dummy_na=False)

# 4. Удаление ненужных столбцов
combined_df.drop(['PassengerId', 'Name', 'Ticket', 'SibSp', 'Parch', 'Cabin'], axis=1, inplace=True)

# Разделяем обратно на обучающую и тестовую выборки
X_train = combined_df[combined_df['is_train'] == 1].drop('is_train', axis=1)
X_test = combined_df[combined_df['is_train'] == 0].drop('is_train', axis=1)
y_train = train_target

# Выравнивание столбцов (важно!)
train_cols = X_train.columns.tolist()
test_cols = X_test.columns.tolist()

missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test[c] = 0

missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    X_train[c] = 0

X_test = X_test[train_cols]
X_train = X_train[train_cols]

print("\n--- X_train после предобработки (первые 5 строк) ---")
print(X_train.head())
print(f"\nРазмер X_train: {X_train.shape}")
print(f"Размер X_test: {X_test.shape}")

# ==============================================================================
# Шаг 3: Обучение и подбор гиперпараметров для RandomForestClassifier
# Это наш основной кандидат
# ==============================================================================

print("\n--- Подбор гиперпараметров для RandomForestClassifier ---")

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) # 10 фолдов для стабильности

rf_model = RandomForestClassifier(random_state=42)

param_grid_rf = {
    'n_estimators': [100, 200, 300, 400],  # Увеличиваем диапазон
    'max_depth': [6, 8, 10, None],  # Ограничиваем, чтобы избежать переобучения
    'min_samples_leaf': [2, 3, 4],
    'min_samples_split': [5, 8, 10]
}

grid_search_rf = GridSearchCV(estimator=rf_model,
                              param_grid=param_grid_rf,
                              cv=kf,
                              scoring='accuracy',
                              n_jobs=-1,
                              verbose=1)

grid_search_rf.fit(X_train, y_train)

print("\n--- Лучшие параметры для RandomForestClassifier ---")
print(grid_search_rf.best_params_)
print(f"Лучшая средняя точность на кросс-валидации (RF): {grid_search_rf.best_score_:.4f}")

best_rf_model = grid_search_rf.best_estimator_

# ==============================================================================
# Шаг 4: Попытка с VotingClassifier (ансамбль моделей)
# Это часто дает лучший результат за счет усреднения
# ==============================================================================

print("\n--- Настройка и обучение VotingClassifier ---")

# Модель 1: Лучший RandomForest
# best_rf_model = RandomForestClassifier(random_state=42, **grid_search_rf.best_params_) # Можно взять параметры вручную

# Модель 2: Logistic Regression (хорошая базовая модель)
log_reg_model = LogisticRegression(random_state=42, solver='liblinear', max_iter=200) # solver='liblinear' хорош для небольших датасетов

# Модель 3: Support Vector Machine (SVC) - мощная, но может быть медленной, поэтому берем с меньшими параметрами
# Или можно взять XGBoost, если он хорошо себя показал
# svc_model = SVC(random_state=42, probability=True, gamma='auto') # probability=True для predict_proba в VotingClassifier

# Модель 3 (Альтернатива SVC - XGBoost):
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss',
                           n_estimators=100, max_depth=4, learning_rate=0.1)


# Создаем VotingClassifier
# weights можно настроить, если одна модель лучше другой, но для начала равные веса
voting_clf = VotingClassifier(estimators=[
    ('rf', best_rf_model),
    ('lr', log_reg_model),
    ('xgb', xgb_model) # Используем XGBoost вместо SVC
], voting='soft', n_jobs=-1) # 'soft' voting использует вероятности, 'hard' - просто большинство голосов

voting_clf.fit(X_train, y_train)

# ==============================================================================
# Шаг 5: Прогнозирование на тестовом наборе и создание файла для отправки
# Используем VotingClassifier как нашу финальную модель
# ==============================================================================

print("\n--- Генерация предсказаний для test.csv с использованием VotingClassifier ---")
final_predictions = voting_clf.predict(X_test)

submission_df = pd.DataFrame({
    'PassengerId': test_passenger_ids,
    'Survived': final_predictions
})

submission_df.to_csv('submission_final.csv', index=False) # Назовем submission_final.csv, чтобы не перезаписать

print("\n--- Первые 5 строк файла предсказаний (submission_final.csv) ---")
print(submission_df.head())
print(f"\nФайл 'submission_final.csv' успешно создан и содержит {len(submission_df)} записей.")
print("Его можно скачать из панели файлов Colab и отправить на проверку.")

# ==============================================================================
# Шаг 6: Анализ важности признаков (от VotingClassifier это сложнее, пока оставим RF)
# ==============================================================================

print("\n--- Важность признаков для предсказания выживаемости (от лучшей RF модели) ---")
feature_importances = pd.Series(best_rf_model.feature_importances_, index=X_train.columns)
sorted_importances = feature_importances.sort_values(ascending=False)

print(sorted_importances)

plt.figure(figsize=(12, 7))
sns.barplot(x=sorted_importances.values, y=sorted_importances.index, palette='viridis')
plt.title('Важность признаков для предсказания выживаемости (лучшая RF модель)')
plt.xlabel('Важность')
plt.ylabel('Признак')
plt.tight_layout()
plt.show()

print("\n--- Конец выполнения программы ---")
