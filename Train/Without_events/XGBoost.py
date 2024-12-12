from sklearn.model_selection import GridSearchCV, cross_val_score
import pandas as pd
from sklearn.metrics import r2_score, make_scorer
from xgboost import XGBRegressor
from Train.preprocess_data import preprocess_data

# Step 1: Load the dataset
file_path = '../../Dataset/data_with_events.csv'

X_scaled, y, kf = preprocess_data(file_path)

# Step 9: Define parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 60, 100],         # Кількість дерев
    'learning_rate': [0.1, 0.15, 0.2],      # Темп навчання
    'max_depth': [7, 8, 9]                  # Глибина дерева
}

xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kf, n_jobs=-1, verbose=1)

# Step 11: Fit the GridSearchCV
grid_search.fit(X_scaled, y)

# Convert results to DataFrame and sort by mean_test_score
results_df = pd.DataFrame(grid_search.cv_results_)
results_df['mean_test_score'] = -results_df['mean_test_score']  # Convert MSE to positive for easier interpretation
sorted_results = results_df.sort_values(by='mean_test_score').reset_index(drop=True)

# Select rows for table: Base model, Intermediate models, Best model, Slightly worse than best
selected_rows = sorted_results.iloc[[0, len(sorted_results) // 4, len(sorted_results) // 2, sorted_results['mean_test_score'].idxmin(), len(sorted_results) - 1]]

# Create table for the article
table = selected_rows[['param_n_estimators', 'param_learning_rate', 'param_max_depth', 'mean_test_score', 'std_test_score']]
table.columns = ['Кількість дерев', 'Темп навчання', 'Глибина дерева', 'Середнє MSE (крос-валідація)', 'Стандартне відхилення MSE']

# Compute R^2 for each selected model
r2_scores = []
for _, row in selected_rows.iterrows():
    model = XGBRegressor(
        n_estimators=row['param_n_estimators'],
        learning_rate=row['param_learning_rate'],
        max_depth=row['param_max_depth'],
        objective='reg:squarederror',
        random_state=42
    )
    r2 = cross_val_score(model, X_scaled, y, cv=kf, scoring=make_scorer(r2_score)).mean()
    r2_scores.append(r2)

table = selected_rows[['param_n_estimators', 'param_learning_rate', 'param_max_depth', 'mean_test_score', 'std_test_score']].copy()
table.columns = ['Кількість дерев', 'Темп навчання', 'Глибина дерева', 'Середнє MSE (крос-валідація)', 'Стандартне відхилення MSE']

# Add R² column to the table
table['R² (крос-валідація)'] = r2_scores

# Display table for the article
print("\nТаблиця результатів:")
print(table)

# Зберегти таблицю результатів у файл CSV
table.to_csv('XGBoost_results.csv', index=False, encoding='utf-8-sig')
