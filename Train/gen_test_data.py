import pandas as pd
from Dataset.additional_functions import *
from datetime import datetime


# Parameters for data generation
products = ['Milk', 'Eggs', 'Chicken', 'Tomatoes', 'Apples', 'Salmon', 'Cheese', 'Lettuce', 'Pork', 'Potatoes']
seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weather_conditions = ['Sunny', 'Rainy', 'Snowy', 'Cloudy', 'Stormy', 'Hot', 'Cold']
events = ['None', 'Birthdays', 'Corporate Event', 'Special Promotion', 'Seasonal Event']


# Generate synthetic data
def generate_test_data():
    data = []
    current_date = datetime(2024, 12, 12)
    season = determine_season(current_date)
    weather = random.choice(seasonal_weather[season])
    num_customers = generate_num_customers(current_date, season, weather)
    sales = get_average_check(num_customers)
    for i in range(len(products)):
        product = products[i]
        year = current_date.year
        day_of_week = days_of_week[current_date.weekday()]
        event = None
        stocks = get_stock(num_customers, event)
        days_until_next_purchase = next_purchase(stocks, product)
        category = get_category(product)
        quantity = determine_quantity(num_customers, stocks, season, product, event)
        unit_price = get_price(season, product, year)
        shelf_life = get_shelf_life(product)

        # Append row data
        data.append([current_date, day_of_week, season, weather, product, category, unit_price,
                     num_customers, sales, stocks, shelf_life, days_until_next_purchase, event, quantity])

    # Create DataFrame
    columns = ['Date', 'Day_of_Week', 'Season', 'Weather', 'Product', 'Category', 'Unit_Price', 'Num_Customers',
               'Sales', 'Stocks', 'Shelf_Life', 'Days_Until_Next_Purchase', 'Event', 'Purchase_Quantity']

    df = pd.DataFrame(data, columns=columns)
    return df


# Generate synthetic data
def generate_test_data_events():
    data = []

    current_date = datetime(2024, 12, 12)
    holiday_name = "Birthdays"

    season = determine_season(current_date)
    weather = random.choice(seasonal_weather[season])
    num_customers = generate_num_customers(current_date, season, weather)
    sales = get_average_check(num_customers)
    for i in range(len(products)):
        product = products[i]
        year = current_date.year
        day_of_week = days_of_week[current_date.weekday()]
        event = holiday_name
        stocks = get_stock(num_customers, event)
        days_until_next_purchase = next_purchase(stocks, product)
        category = get_category(product)
        quantity = determine_quantity(num_customers, stocks, season, product, event)
        unit_price = get_price(season, product, year)
        shelf_life = get_shelf_life(product)

        # Append row data
        data.append([current_date, day_of_week, season, weather, product, category, unit_price,
                     num_customers, sales, stocks, shelf_life, days_until_next_purchase, event, quantity])

    # Create DataFrame
    columns = ['Date', 'Day_of_Week', 'Season', 'Weather', 'Product', 'Category', 'Unit_Price', 'Num_Customers',
               'Sales', 'Stocks', 'Shelf_Life', 'Days_Until_Next_Purchase', 'Event', 'Purchase_Quantity']

    df = pd.DataFrame(data, columns=columns)
    return df


# Generate synthetic data
synthetic_data = generate_test_data_events()

# To display the first 5 rows of the DataFrame
print(synthetic_data.head())

# Or, if you'd like to export it to a CSV file
synthetic_data.to_csv('10_rows.csv', index=False)
