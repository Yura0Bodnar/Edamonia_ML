import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
import numpy as np

# Step 1: Load the dataset
df = pd.read_csv('../../Dataset/synthetic_data.csv')

# Step 2: Convert 'Date' into separate Year, Month, Day columns
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# Step 3: Label encode 'Product'
label_encoder_product = LabelEncoder()
df['Product_Label'] = label_encoder_product.fit_transform(df['Product'])

# Step 4: OneHot encode other categorical columns
onehot_encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_columns = onehot_encoder.fit_transform(df[['Day_of_Week', 'Season', 'Weather', 'Category']])
encoded_column_names = onehot_encoder.get_feature_names_out(['Day_of_Week', 'Season', 'Weather', 'Category'])
encoded_df = pd.DataFrame(encoded_columns, columns=encoded_column_names)

# Step 5: Concatenate the original DataFrame with the encoded DataFrame
df = pd.concat([df, encoded_df], axis=1)

# Step 6: Drop the original categorical columns and 'Date'
df = df.drop(['Day_of_Week', 'Season', 'Weather', 'Product', 'Date', 'Category'], axis=1)

# Step 7: Split features and target
X = df.drop(['Purchase_Quantity'], axis=1)  # Features
y = df['Purchase_Quantity']  # Target

# Step 8: Standardize the numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 9: Set up cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation

# Define the model
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)

# Define scoring metrics for cross-validation
scoring = {
    'MSE': make_scorer(mean_squared_error),
    'MAE': make_scorer(mean_absolute_error),
    'R2': make_scorer(r2_score)
}

# Perform cross-validation
cv_results = {}
for metric, scorer in scoring.items():
    scores = cross_val_score(xgb_model, X_scaled, y, cv=kf, scoring=scorer)
    cv_results[metric] = scores
    print(f"{metric} (5-Fold Cross-Validation): {scores.mean():.4f} ± {scores.std():.4f}")

# After cross-validation, train the model on the entire training set for final evaluation on a test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
xgb_model.fit(X_train, y_train)

# Make predictions and evaluate on the test set
y_test_pred = xgb_model.predict(X_test)

# Calculate evaluation metrics for the test set
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)  # Root Mean Squared Error
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)  # R-squared

print("\nTest Set Metrics:")
print(f"Mean Squared Error (MSE): {test_mse}")
print(f"Root Mean Squared Error (RMSE): {test_rmse}")
print(f"Mean Absolute Error (MAE): {test_mae}")
print(f"R-squared (R²): {test_r2}")
