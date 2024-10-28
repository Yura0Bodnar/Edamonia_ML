import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Step 1: Load the dataset
df = pd.read_csv('../synthetic_purchase_data.csv')

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
df = df.drop(['Day_of_Week', 'Season', 'Weather', 'Product', 'Date', 'Category'], axis=1)  # add event

# Step 7: Split features and target
X = df.drop(['Purchase_Quantity'], axis=1)  # Features
y = df['Purchase_Quantity']  # Target

# Step 8: Standardize the numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 9: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 10: Train a Decision Tree model
tree_model = DecisionTreeRegressor(random_state=42, max_depth=5, min_samples_leaf=10)
tree_model.fit(X_train, y_train)

# Step 11: Make predictions and evaluate
y_pred = tree_model.predict(X_test)

# Step 12: Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # Root Mean Squared Error
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)  # R-squared

# Print evaluation metrics
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R²): {r2}")
