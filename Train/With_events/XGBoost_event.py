from sklearn.model_selection import GridSearchCV, cross_val_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold
from Train.preprocess_data import preprocess_data_event

# Step 1: Load the dataset
file_path = '../../Dataset/data_with_events.csv'

X_scaled, y, kf = preprocess_data_event(file_path)

# Step 9: Define parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 60, 100],         # Number of trees
    'learning_rate': [0.1, 0.15, 0.2],      # Learning rate
    'max_depth': [7, 8, 9]                  # Tree depth
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
table.columns = ['Number of trees', 'Learning rate', 'Tree depth', 'Mean MSE (cross-validation)', 'Std deviation MSE']

# Compute R² for each selected model
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

table['R² (cross-validation)'] = r2_scores

# Display table for the article
print("\nResults table:")
print(table)

# Save the results table to CSV
table.to_csv('XGBoost_results_w_e.csv', index=False, encoding='utf-8-sig')

# Step 15: Add predictions back to the original test set
# Load the original dataset
original_df = pd.read_csv(file_path)

# Split into train/test (assuming X_test is provided)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the best model on the full training data
best_model = grid_search.best_estimator_

# Predict with the best estimator on the test set
y_test_pred = best_model.predict(X_test)

# Get test indices (assuming the order is preserved after preprocessing)
test_indices = original_df.index[len(X_train):]

# Add predictions to a copy of the original test set
output_df = original_df.iloc[test_indices].copy()
output_df['Predicted_Purchase_Quantity'] = y_test_pred

# Step 16: Save predictions with original columns
output_df.to_csv('XGBoost_predict_w_e.csv', index=False, encoding='utf-8-sig')

# Display example predictions
print("\nPredictions with original columns:")
print(output_df.head())
