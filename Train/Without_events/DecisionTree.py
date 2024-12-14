import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
import numpy as np
from Train.preprocess_data import preprocess_data

# Step 1: Load the dataset
file_path = '../../Dataset/synthetic_data.csv'

X_scaled, y, kf = preprocess_data(file_path)

# Step 2: Define parameter grid for GridSearchCV
param_grid = {
    'max_depth': [15, 16],        # Глибина дерева
    'min_samples_split': [15, 16, 17, 18, 19],    # Мінімальна кількість зразків для розбиття вузла
    'min_samples_leaf': [14, 15, 16, 17, 18]       # Мінімальна кількість зразків у листі
}

tree_model = DecisionTreeRegressor(random_state=42)
grid_search = GridSearchCV(
    estimator=tree_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=kf,
    n_jobs=-1,
    verbose=1
)

# Step 3: Fit the GridSearchCV
grid_search.fit(X_scaled, y)

# Step 4: Convert results to DataFrame and sort by mean_test_score
results_df = pd.DataFrame(grid_search.cv_results_)
results_df['mean_test_score'] = -results_df['mean_test_score']  # Convert MSE to positive for easier interpretation
sorted_results = results_df.sort_values(by='mean_test_score').reset_index(drop=True)

# Step 5: Select rows for table
selected_rows = sorted_results.iloc[
    [0, len(sorted_results) // 4, len(sorted_results) // 2, sorted_results['mean_test_score'].idxmin(), len(sorted_results) - 1]
]

# Step 6: Create table for the article
table = selected_rows[['param_max_depth', 'param_min_samples_split', 'param_min_samples_leaf', 'mean_test_score', 'std_test_score']].copy()
table.columns = ['Глибина дерева', 'Мінімальні зразки для розбиття', 'Мінімальні зразки в листі', 'Середнє MSE (крос-валідація)', 'Стандартне відхилення MSE']

# Step 7: Compute R² for each selected model
r2_scores = []
for _, row in selected_rows.iterrows():
    model = DecisionTreeRegressor(
        max_depth=row['param_max_depth'],
        min_samples_split=row['param_min_samples_split'],
        min_samples_leaf=row['param_min_samples_leaf'],
        random_state=42
    )
    r2 = cross_val_score(model, X_scaled, y, cv=kf, scoring=make_scorer(r2_score)).mean()
    r2_scores.append(r2)

# Step 8: Add R² column to the table
table['R² (крос-валідація)'] = r2_scores

# Step 9: Save table to CSV
table.to_csv('DecisionTree_results.csv', index=False, encoding='utf-8-sig')

# Step 10: Display table
print("\nТаблиця результатів:")
print(table)

# Step 11: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 12: Train the best model on training data
best_params = grid_search.best_params_
best_model = DecisionTreeRegressor(**best_params, random_state=42)
best_model.fit(X_train, y_train)

# Step 13: Predict on the test data
y_test_pred = best_model.predict(X_test)

# Step 14: Evaluate predictions
mse = mean_squared_error(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print(f"\nОцінка моделі на тестових даних:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Step 15: Add predictions back to the original test set
# Load the original dataset
original_df = pd.read_csv(file_path)
test_indices = original_df.index[len(X_train):]  # Assuming the order is preserved
output_df = original_df.iloc[test_indices].copy()
output_df['Predicted_Purchase_Quantity'] = y_test_pred

# Step 16: Save predictions with original columns
output_df.to_csv('DecisionTree_predict.csv', index=False, encoding='utf-8-sig')

# Display example predictions
print("\nРезультат з оригінальними колонками і предиктами:")
print(output_df.head())
