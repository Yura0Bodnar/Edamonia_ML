import matplotlib.pyplot as plt
import numpy as np

# Імпорт найнижчих значень MSE для кожної моделі
from LinearRegression import linear_mse_value
from XGBoost import xgboost_mse_value
from DecisionTree import tree_mse_value
from CatBoost import catboost_mse_value
from LightGBM import lightgbm_mse_value

# Дані для графіка
models = ['Linear Regression', 'XGBoost', 'Decision Tree', 'CatBoost', 'LightGBM']
mse_values = [linear_mse_value, xgboost_mse_value, tree_mse_value, catboost_mse_value, lightgbm_mse_value]

# Створення графіка
plt.figure(figsize=(10, 6))
plt.bar(models, mse_values, color='skyblue', edgecolor='black')

# Додавання підписів
plt.title('Comparison of MSE Values for Different Models', fontsize=16)
plt.ylabel('Mean Squared Error (MSE)', fontsize=14)
plt.xlabel('Models', fontsize=14)
plt.xticks(rotation=15, fontsize=12)
plt.yticks(fontsize=12)

# Додавання значень над стовпцями
for i, value in enumerate(mse_values):
    plt.text(i, value + 0.02, f'{value:.4f}', ha='center', fontsize=10)

# Відображення графіка
plt.tight_layout()
plt.show()