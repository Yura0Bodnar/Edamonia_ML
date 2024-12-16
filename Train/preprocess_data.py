import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold

# Step 1: Combine Rare Categories
def group_events(event):
    holidays = [
        "New Year", "Women's Day", "Men's Day", "Independence Day of Ukraine",
        "Constitution Day of Ukraine", "Day of Defenders of Ukraine",
        "Valentine's Day", "Teacher's Day", "Day of Lviv city",
        "Day of Dignity and Freedom", "Day of Ukrainian Language",
        "The Nativity of Christ", "Saint Nicholas Day", "Easter"
    ]
    promotions = ["Special Promotion", "Seasonal Event"]
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
        return event


def preprocess_data_event(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Convert 'Date' into separate Year, Month, Day columns
    df['Date'] = pd.to_datetime(df['Date'])
    df[['Year', 'Month', 'Day']] = df['Date'].apply(lambda x: [x.year, x.month, x.day]).to_list()

    # Helper function for OneHot Encoding
    def onehot_encode(df, columns, prefix):
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        encoded = encoder.fit_transform(df[columns])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(prefix), index=df.index)
        return encoded_df

    # OneHot encode 'Product'
    product_encoded_df = onehot_encode(df, ['Product'], ['Product'])

    # OneHot encode other categorical columns
    categorical_columns = ['Day_of_Week', 'Season', 'Weather', 'Category']
    categorical_encoded_df = onehot_encode(df, categorical_columns, categorical_columns)

    # Concatenate all encoded data
    df = pd.concat([df, product_encoded_df, categorical_encoded_df], axis=1)

    # Drop original categorical columns and 'Date'
    df = df.drop(['Day_of_Week', 'Season', 'Weather', 'Product', 'Date', 'Category'], axis=1)

    # Group and encode 'Event'
    df['Event_Grouped'] = df['Event'].apply(group_events)
    event_encoded_df = onehot_encode(df, ['Event_Grouped'], ['Event_Grouped'])

    # Concatenate the encoded Event Data
    df = pd.concat([df, event_encoded_df], axis=1)

    # Drop the original 'Event' and grouped column
    df = df.drop(['Event', 'Event_Grouped'], axis=1)

    # Split features and target
    X = df.drop(['Purchase_Quantity'], axis=1)  # Features
    y = df['Purchase_Quantity']  # Target

    # Standardize the numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Set up cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation

    return X_scaled, y, kf


def preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Convert 'Date' into separate Year, Month, Day columns
    df['Date'] = pd.to_datetime(df['Date'])
    df[['Year', 'Month', 'Day']] = df['Date'].apply(lambda x: [x.year, x.month, x.day]).to_list()

    # Helper function for OneHot Encoding
    def onehot_encode(df, columns, prefix):
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        encoded = encoder.fit_transform(df[columns])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(prefix), index=df.index)
        return encoded_df

    # OneHot encode 'Product'
    product_encoded_df = onehot_encode(df, ['Product'], ['Product'])

    # OneHot encode other categorical columns
    categorical_columns = ['Day_of_Week', 'Season', 'Weather', 'Category']
    categorical_encoded_df = onehot_encode(df, categorical_columns, categorical_columns)

    # Concatenate all encoded data
    df = pd.concat([df, product_encoded_df, categorical_encoded_df], axis=1)

    # Drop original categorical columns and 'Date'
    columns_to_drop = ['Day_of_Week', 'Season', 'Weather', 'Product', 'Date', 'Category']
    if 'Event' in df.columns:
        columns_to_drop.append('Event')

    df = df.drop(columns=columns_to_drop)

    # Split features and target
    X = df.drop(['Purchase_Quantity'], axis=1)  # Features
    y = df['Purchase_Quantity']  # Target

    # Standardize the numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Set up cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation

    return X_scaled, y, kf
