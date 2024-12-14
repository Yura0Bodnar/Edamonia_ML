import importlib

# Введення даних
date = input("Введіть дату (у форматі YYYY-MM-DD): ")

# Чи є івент
events = int(input("Чи є Events? (1 для так, 0 для ні): ").strip())

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

# Перевірка, чи є така модель
if model_name not in model_name_mapping:
    print("Помилка: невідома модель.")
    exit()

# Коректне ім'я модуля
model_class_name = model_name_mapping[model_name]

try:
    # Визначення шляху до модуля
    if events == 0:
        module_path = f"Train.Without_events.{model_class_name}"
    elif events == 1:
        module_path = f"Train.With_events.{model_class_name}_event"
    else:
        print("Помилка: некоректне значення для Events.")
        exit()

    # Динамічний імпорт модуля
    module = importlib.import_module(module_path)


except ModuleNotFoundError:
    print(f"Помилка: модуль '{model_class_name}' не знайдено.")
except Exception as e:
    print(f"Помилка під час виконання моделі: {e}")
