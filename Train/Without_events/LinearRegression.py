from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
import numpy as np
from Train.preprocess_data import preprocess_data
import pandas as pd

# Step 1: Load the dataset
file_path = '../../Dataset/data_with_events.csv'

X_scaled, y, kf = preprocess_data(file_path)

# Step 2: Define the Linear Regression model
lin_reg_model = LinearRegression()

# Step 3: Define scoring metrics for cross-validation
scoring = {
    'MSE': make_scorer(mean_squared_error),
    'MAE': make_scorer(mean_absolute_error),
    'R2': make_scorer(r2_score)
}

# Step 4: Perform cross-validation for each metric
cv_results = {}
for metric, scorer in scoring.items():
    scores = cross_val_score(lin_reg_model, X_scaled, y, cv=kf, scoring=scorer)
    cv_results[metric] = scores
    print(f"{metric} (5-Fold Cross-Validation): {scores.mean():.4f} ± {scores.std():.4f}")

# Step 5: Save cross-validation results to a DataFrame
cv_results_df = pd.DataFrame({
    'Metric': list(cv_results.keys()),
    'Mean Score': [np.mean(scores) for scores in cv_results.values()],
    'Std Dev': [np.std(scores) for scores in cv_results.values()]
})

# Step 6: Save cross-validation results to CSV
cv_results_df.to_csv('LinearRegression_CV_Results.csv', index=False, encoding='utf-8-sig')

# Step 7: Train the model on the entire training set for final evaluation
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
lin_reg_model.fit(X_train, y_train)

# Step 8: Make predictions and evaluate on the test set
y_test_pred = lin_reg_model.predict(X_test)

# Step 9: Calculate evaluation metrics for the test set
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Step 10: Save test set results to a DataFrame
test_results_df = pd.DataFrame({
    'Metric': ['MSE', 'RMSE', 'MAE', 'R2'],
    'Value': [test_mse, test_rmse, test_mae, test_r2]
})

# Step 11: Save test set results to CSV
test_results_df.to_csv('LinearRegression_Test_Results.csv', index=False, encoding='utf-8-sig')

# Step 12: Display results
print("\nCross-Validation Results:")
print(cv_results_df)
print("\nTest Set Metrics:")
print(test_results_df)

linear_mse_value = test_results_df.loc[test_results_df['Metric'] == 'MSE', 'Value'].values[0]
print(f"Test MSE: {linear_mse_value}")
