from sklearn.model_selection import GridSearchCV, cross_val_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold

# Step 1: Load the dataset
df = pd.read_csv('../../Dataset/data_with_events.csv')

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

# Step 1: Combine Rare Categories
def group_events(event):
    holidays = [
        "New Year", "Women's Day", "Men's Day", "Independence Day of Ukraine",
        "Constitution Day of Ukraine", "Day of Defenders of Ukraine",
        "Valentine's Day", "Teacher's Day", "Day of Lviv city",
        "Day of Dignity and Freedom", "Day of Ukrainian Language",
        "The Nativity of Christ", "Saint Nicholas Day", "Easter"
    ]
    promotions = ["Holiday Special", "Special Promotion", "Seasonal Event"]
    occasions = ["Birthdays", "Corporate Event"]

    if event in holidays:
        return "Holiday"
    elif event in promotions:
        return "Promotion"
    elif event in occasions:
        return "Occasion"
    elif event == "None":
        return "None"
    else:
        return "Other"


df['Event_Grouped'] = df['Event'].apply(group_events)
# Step 2: OneHotEncode the Grouped Column
onehot_encoder_event = OneHotEncoder(drop='first', sparse_output=False)
event_encoded_columns = onehot_encoder_event.fit_transform(df[['Event_Grouped']])
event_encoded_column_names = onehot_encoder_event.get_feature_names_out(['Event_Grouped'])
event_encoded_df = pd.DataFrame(event_encoded_columns, columns=event_encoded_column_names)

# Step 3: Concatenate the Encoded Data
df = pd.concat([df, event_encoded_df], axis=1)

# Step 4: Drop the Original 'Event' and Grouped Column
df = df.drop(['Event', 'Event_Grouped'], axis=1)

# Step 7: Split features and target
X = df.drop(['Purchase_Quantity'], axis=1)  # Features
y = df['Purchase_Quantity']  # Target

# Step 8: Standardize the numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 9: Define parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 60, 100],         # Кількість дерев
    'learning_rate': [0.1, 0.15, 0.2],      # Темп навчання
    'max_depth': [7, 8, 9]                  # Глибина дерева
}

# Step 10: Set up cross-validation and GridSearchCV
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation
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
table.to_csv('results_table.csv', index=False, encoding='utf-8-sig')
