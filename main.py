import importlib
from Train.gen_test_data import generate_test_data
import os

# Введення даних
date = input("Введіть дату (у форматі YYYY-MM-DD): ")

# Чи є івент
events = int(input("Чи є Events? (0 для ні, 1 для Holiday, 2 для Daily event, 3 для Promotion): ").strip())

if events == 0:
    test_data = generate_test_data(date, 0)
elif events == 1:
    test_data = generate_test_data(date, 1)
elif events == 2:
    test_data = generate_test_data(date, 2)
else:
    test_data = generate_test_data(date, 3)

dataset_path = os.path.abspath("Dataset")
test_data.to_csv(f"{dataset_path}/10_rows.csv", index=False)

# Яку модель беремо
model_name = input("Введіть назву моделі (catboost, decision_tree, lightgbm, linearregression, xgboost): ").strip().lower()

# Маппінг для коректної капіталізації назв
model_name_mapping = {
    "catboost": "CatBoost",
    "decision_tree": "DecisionTree",
    "lightgbm": "LightGBM",
    "linearregression": "LinearRegression",
    "xgboost": "XGBoost"
}

# Коректне ім'я модуля
if model_name in model_name_mapping:
    model_class_name = model_name_mapping[model_name]
    module_path = f"Train.Predict.{model_class_name}"  # Правильний шлях до файлу
    try:
        # Імпортуємо файл динамічно
        module = importlib.import_module(module_path)

        # Викликаємо функцію train та передаємо параметри
        print(f"Запуск моделі '{model_class_name}'...")
        module.train(events, os.path.abspath("Dataset"))  # Передаємо events і абсолютний шлях до Dataset

    except ModuleNotFoundError:
        print(f"Помилка: файл '{model_class_name}.py' не знайдено у директорії Train/Predict.")
    except Exception as e:
        print(f"Помилка під час виконання моделі: {e}")
else:
    print("Помилка: введена некоректна назва моделі.")
