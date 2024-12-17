from sklearn.model_selection import GridSearchCV, cross_val_score
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from Train.preprocess_data import preprocess_data, preprocess_test_data

def train(events, dataset_path):

    if events == 0:
        # Step 1: Load the dataset
        file_path = f"{dataset_path}/synthetic_data.csv"

        # Preprocessing data
        X_scaled, y, kf = preprocess_data(file_path, 1)

        param_grid = {
            'n_estimators': [50, 60, 100],  # Кількість дерев
            'learning_rate': [0.1, 0.15, 0.2],  # Темп навчання
            'max_depth': [7, 8, 9]  # Глибина дерева
        }
    else:
        file_path = f"{dataset_path}/data_with_events.csv"

        # Preprocessing data
        X_scaled, y, kf = preprocess_data(file_path, 1)

        param_grid = {
            'n_estimators': [50, 60, 100],  # Number of trees
            'learning_rate': [0.1, 0.15, 0.2],  # Learning rate
            'max_depth': [7, 8, 9]  # Tree depth
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

    # Save the results table to CSV
    table.to_csv('Train/Results/XGBoost_results.csv', index=False, encoding='utf-8-sig')

    # Step 11: Make predictions using the best model from GridSearchCV
    best_model = grid_search.best_estimator_
    # Step 7: Get the best model's parameters
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_

    # Step 8: Print the best model parameters
    print("\nПараметри найкращої моделі XGBoost:")
    print(f"Кількість ітерацій: {best_params['n_estimators']}")
    print(f"Темп навчання: {best_params['learning_rate']}")
    print(f"Максимальна глибина дерева: {best_params['max_depth']}")
    print(f"Середнє MSE (крос-валідація): {best_score:.4f}")

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

    custom_test_file = f"{dataset_path}/10_rows.csv"
    custom_test_data = pd.read_csv(custom_test_file).copy()

    X_custom, y_custom = preprocess_test_data(custom_test_file, events)

    # Step 15: Make predictions on the custom test table
    custom_test_predictions = best_model.predict(X_custom)

    custom_test_data.loc[:, 'Прогноз'] = custom_test_predictions
    custom_test_data.to_csv('Train/Results/XGBoost_predict.csv', index=False, encoding='utf-8-sig')

    # Step 18: Display the updated custom test table
    print("\nТаблиця з прогнозами:")
    print(custom_test_data.head())
