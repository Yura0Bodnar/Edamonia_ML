import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold


def preprocess_data_event(file_path):
    # Step 1: Load the dataset
    df = pd.read_csv(file_path)

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

    # Step 9: Set up cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation

    return X_scaled, y, kf


def preprocess_data(file_path):
    # Step 1: Load the dataset
    df = pd.read_csv(file_path)

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
    columns_to_drop = ['Day_of_Week', 'Season', 'Weather', 'Product', 'Date', 'Category']
    if 'Event' in df.columns:
        columns_to_drop.append('Event')

    df = df.drop(columns=columns_to_drop)

    # Step 7: Split features and target
    X = df.drop(['Purchase_Quantity'], axis=1)  # Features
    y = df['Purchase_Quantity']  # Target

    # Step 8: Standardize the numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 9: Set up cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation

    return X_scaled, y, kf
