from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
import numpy as np
from Train.preprocess_data import preprocess_data_event
import pandas as pd

# Step 1: Load the dataset
file_path = '../../Dataset/data_with_events.csv'

X_scaled, y, kf = preprocess_data_event(file_path)

# Step 2: Define parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],       # Кількість дерев
    'learning_rate': [0.05, 0.1, 0.2],    # Темп навчання
    'max_depth': [3, 5, 7],               # Глибина дерева
    'num_leaves': [15, 31, 50]            # Кількість листків
}

# Step 3: Define LightGBM model
lgbm_model = LGBMRegressor(objective='regression', random_state=42, verbose=-1)

# Step 4: Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=lgbm_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',  # Оптимізація за MSE
    cv=kf,
    n_jobs=-1,
    verbose=1
)

# Step 5: Fit GridSearchCV
grid_search.fit(X_scaled, y)

# Step 6: Convert results to DataFrame and sort by mean_test_score
results_df = pd.DataFrame(grid_search.cv_results_)
results_df['mean_test_score'] = -results_df['mean_test_score']  # Convert MSE to positive for easier interpretation
sorted_results = results_df.sort_values(by='mean_test_score').reset_index(drop=True)

# Step 7: Select rows for the table
selected_rows = sorted_results.iloc[
    [0, len(sorted_results) // 4, len(sorted_results) // 2, sorted_results['mean_test_score'].idxmin(), len(sorted_results) - 1]
]

# Step 8: Create table for the article
table = selected_rows[
    ['param_n_estimators', 'param_learning_rate', 'param_max_depth', 'param_num_leaves', 'mean_test_score', 'std_test_score']
].copy()
table.columns = [
    'Кількість дерев', 'Темп навчання', 'Глибина дерева', 'Кількість листків',
    'Середнє MSE (крос-валідація)', 'Стандартне відхилення MSE'
]

# Step 9: Compute R² for each selected model
r2_scores = []
for _, row in selected_rows.iterrows():
    model = LGBMRegressor(
        n_estimators=row['param_n_estimators'],
        learning_rate=row['param_learning_rate'],
        max_depth=row['param_max_depth'],
        num_leaves=row['param_num_leaves'],
        objective='regression',
        random_state=42,
        verbose=-1
    )
    r2 = cross_val_score(model, X_scaled, y, cv=kf, scoring=make_scorer(r2_score)).mean()
    r2_scores.append(r2)

# Step 10: Add R² column to the table
table['R² (крос-валідація)'] = r2_scores

# Step 11: Save table to CSV
table.to_csv('LightGBM_results_w_e.csv', index=False, encoding='utf-8-sig')

# Step 12: Display table
print("\nТаблиця результатів:")
print(table)

# Step 13: Train and evaluate the best model
best_model = grid_search.best_estimator_

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
best_model.fit(X_train, y_train)

y_test_pred = best_model.predict(X_test)

# Step 14: Calculate evaluation metrics for the test set
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\nTest Set Metrics:")
print(f"Mean Squared Error (MSE): {test_mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {test_rmse:.4f}")
print(f"Mean Absolute Error (MAE): {test_mae:.4f}")
print(f"R-squared (R²): {test_r2:.4f}")

# Step 15: Add predictions back to the original test set
original_df = pd.read_csv(file_path)
test_indices = original_df.index[len(X_train):]  # Assuming the order is preserved
output_df = original_df.iloc[test_indices].copy()
output_df['Predicted_Purchase_Quantity'] = y_test_pred

# Step 16: Save predictions with original columns
output_df.to_csv('LightGBM_predict_w_e.csv', index=False, encoding='utf-8-sig')

# Display example predictions
print("\nРезультат з оригінальними колонками і предиктами:")
print(output_df.head())
