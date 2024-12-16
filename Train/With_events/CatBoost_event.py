from Train.preprocess_data import preprocess_data_event
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import pandas as pd
from Train.preprocess_data import group_events

# Step 1: Load the dataset
file_path = '../../Dataset/data_with_events.csv'

X_scaled, y, kf = preprocess_data_event(file_path)

# Step 2: Define parameter grid for GridSearchCV
param_grid = {
    'iterations': [100, 200, 300],         # Кількість ітерацій
    'learning_rate': [0.05, 0.1, 0.2],    # Темп навчання
    'depth': [4, 6, 8]                    # Глибина дерева
}

# Step 3: Define the CatBoost model
catboost_model = CatBoostRegressor(verbose=0, random_state=42)

# Step 4: Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=catboost_model,
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
selected_rows = sorted_results.head(5)  # Top 5 results

# Step 8: Create table for the article
table = selected_rows[
    ['param_iterations', 'param_learning_rate', 'param_depth', 'mean_test_score', 'std_test_score']
].copy()
table.columns = [
    'Кількість ітерацій', 'Темп навчання', 'Глибина дерева',
    'Середнє MSE (крос-валідація)', 'Стандартне відхилення MSE'
]

# Step 9: Compute R² for each selected model
r2_scores = []
for _, row in selected_rows.iterrows():
    model = CatBoostRegressor(
        iterations=row['param_iterations'],
        learning_rate=row['param_learning_rate'],
        depth=row['param_depth'],
        verbose=0,
        random_state=42
    )
    r2 = cross_val_score(model, X_scaled, y, cv=kf, scoring=make_scorer(r2_score)).mean()
    r2_scores.append(r2)

# Step 10: Add R² column to the table
table['R² (крос-валідація)'] = r2_scores

# Step 11: Save table to CSV
table.to_csv('CatBoost_results.csv', index=False, encoding='utf-8-sig')

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

# New function to preprocess test dataset
def preprocess_test_data(file_path):
    # Step 1: Load the dataset
    df = pd.read_csv(file_path)

    # Step 2: Convert 'Date' into separate Year, Month, Day columns
    df['Date'] = pd.to_datetime(df['Date'])
    df[['Year', 'Month', 'Day']] = df['Date'].apply(lambda x: [x.year, x.month, x.day]).to_list()

    def onehot_encode(df, columns, prefix):
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        encoded = encoder.fit_transform(df[columns])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(prefix), index=df.index)
        return encoded_df

    # Step 3: Label encode 'Product'
    product_encoded_df = onehot_encode(df, ['Product'], ['Product'])

    # Step 4: OneHot encode other categorical columns
    train_categories = {
        'Day_of_Week': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        'Season': ['Winter', 'Spring', 'Summer', 'Autumn'],
        'Weather': ['Sunny', 'Rainy', 'Snowy', 'Cloudy', 'Stormy', 'Hot', 'Cold'],
        'Category': ['Dairy', 'Meat', 'Vegetables', 'Fruits', 'Seafood']
    }
    train_onehot_encoder = OneHotEncoder(categories=list(train_categories.values()), drop='first', sparse_output=False)
    encoded_columns = train_onehot_encoder.fit_transform(df[['Day_of_Week', 'Season', 'Weather', 'Category']])
    encoded_column_names = train_onehot_encoder.get_feature_names_out(['Day_of_Week', 'Season', 'Weather', 'Category'])
    encoded_df = pd.DataFrame(encoded_columns, columns=encoded_column_names, index=df.index)

    # Step 5: Concatenate the original DataFrame with the encoded DataFrame
    df = pd.concat([df, product_encoded_df, encoded_df], axis=1)

    # Step 6: Drop the original categorical columns and 'Date'
    df = df.drop(['Day_of_Week', 'Season', 'Weather', 'Product', 'Date', 'Category'], axis=1)

    # Step 7: Group 'Event' and one-hot encode it
    holidays = [
        "New Year", "Women's Day", "Men's Day", "Independence Day of Ukraine",
        "Constitution Day of Ukraine", "Day of Defenders of Ukraine",
        "Valentine's Day", "Teacher's Day", "Day of Lviv city",
        "Day of Dignity and Freedom", "Day of Ukrainian Language",
        "The Nativity of Christ", "Saint Nicholas Day", "Easter"
    ]
    promotions = ["Special Promotion", "Seasonal Event"]
    occasions = ["Birthdays", "Corporate Event"]

    # Dynamically generate all possible grouped categories
    all_possible_events = list(set(
        ["Holiday" if e in holidays else
         "Promotion" if e in promotions else
         "Occasion" if e in occasions else
         "None" if e == "None" else e
         for e in holidays + promotions + occasions + ["None"]]
    ))

    df['Event_Grouped'] = df['Event'].apply(group_events)
    # Step 4: OneHot encode the 'Event_Grouped' column
    encoder = OneHotEncoder(categories=[all_possible_events], drop='first', sparse_output=False)
    encoded = encoder.fit_transform(df[['Event_Grouped']])
    encoded_columns = encoder.get_feature_names_out(['Event_Grouped'])
    event_encoded_df = pd.DataFrame(encoded, columns=encoded_columns, index=df.index)

    # Step 5: Concatenate the Encoded Data
    df = pd.concat([df, event_encoded_df], axis=1)

    # Step 9: Drop the Original 'Event' and Grouped Column
    df = df.drop(['Event', 'Event_Grouped'], axis=1)
    # Step 10: Extract target variable
    y = df['Purchase_Quantity']
    df = df.drop(['Purchase_Quantity'], axis=1)

    # Step 11: Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    return X_scaled, y

 # Load custom test dataset
custom_test_file = '../../Train/10_rows.csv'
custom_test_data = pd.read_csv(custom_test_file)

X_custom, y_custom = preprocess_test_data(custom_test_file)

model_features = best_model.feature_names_

# Step 15: Make predictions on the custom test table
custom_test_predictions = best_model.predict(X_custom)

# Step 16: Add predictions to the custom test table
custom_test_data['Прогноз'] = custom_test_predictions
# Step 17: Save the updated table with predictions
custom_test_data.to_csv('custom_test_table_with_predictions.csv', index=False, encoding='utf-8-sig')

# # Step 18: Display the updated custom test table
print("\nТаблиця з прогнозами:")
print(custom_test_data.head())

# # Step 15: Add predictions back to the original test set
# # Load the original dataset
# original_df = pd.read_csv(file_path)
#
# # Get test indices (assuming the order is preserved after preprocessing)
# test_indices = original_df.index[len(X_train):]
#
# # Add predictions to a copy of the original test set
# output_df = original_df.iloc[test_indices].copy()
# output_df['Predicted_Value'] = y_test_pred
#
# # Step 16: Save predictions with original columns
# output_df.to_csv('CatBoost_predict_w_e.csv', index=False, encoding='utf-8-sig')
#
# # Display example predictions
# print("\nРезультат з оригінальними колонками і предиктами:")
# print(output_df.head())


