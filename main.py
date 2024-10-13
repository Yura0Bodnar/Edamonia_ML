import random
import pandas as pd
from additional_functions import get_price, get_category
from datetime import timedelta, datetime

# Parameters for data generation
products = ['Milk', 'Eggs', 'Chicken', 'Tomatoes', 'Apples', 'Fish', 'Cheese', 'Lettuce', 'Beef', 'Potatoes']
seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weather_conditions = ['Sunny', 'Rainy', 'Snowy', 'Cloudy', 'Stormy', 'Hot', 'Cold']
events = ['None', 'Holiday Special', 'Discount', 'Corporate Event', 'Special Promotion', 'Seasonal Event']


# Helper function to generate random date
def random_date(start, end):
    return start + timedelta(days=random.randint(0, (end - start).days))


# Generate synthetic data
def generate_synthetic_data(n_rows):
    data = []

    # Date range for purchases
    start_date = datetime(2015, 1, 1)
    end_date = datetime(2024, 10, 1)

    for _ in range(n_rows):
        date = random_date(start_date, end_date)
        product = random.choice(products)
        category = get_category(product)  # Automatically get the correct category
        quantity = random.randint(1, 100)  # Purchase quantity
        days_until_next_purchase = random.randint(1, 30)  # Days until next purchase
        season = random.choice(seasons)
        unit_price = get_price(season)
        day_of_week = days_of_week[date.weekday()]
        holiday = random.choice([0, 1])  # 0 = No, 1 = Yes
        sales = round(random.uniform(500, 10000), 2)  ############## Daily sales in hryvnia
        num_customers = random.randint(50, 500)  # Number of customers in the restaurant
        weather = random.choice(weather_conditions)
        event = random.choice(events)
        stock_left = random.randint(0, 200)  # Remaining stock
        day_type = 'Weekend' if day_of_week in ['Saturday', 'Sunday'] else 'Weekday'
        shelf_life = random.randint(1, 30)  # Shelf life in days

        # Append row data
        data.append([date, product, category, quantity, unit_price, days_until_next_purchase, season, day_of_week,
                     holiday, sales, num_customers, weather, event, stock_left, day_type, shelf_life])

    # Create DataFrame
    columns = ['Date', 'Product', 'Category', 'Purchase_Quantity', 'Unit_Price', 'Days_Until_Next_Purchase', 'Season',
               'Day_of_Week', 'Holiday', 'Sales', 'Num_Customers', 'Weather', 'Event', 'Stock_Left', 'Day_Type',
               'Shelf_Life']

    df = pd.DataFrame(data, columns=columns)
    return df


# Generate synthetic data
synthetic_data = generate_synthetic_data(1000)

# To display the first 5 rows of the DataFrame
print(synthetic_data.head())

# Or, if you'd like to export it to a CSV file
synthetic_data.to_csv('synthetic_purchase_data.csv', index=False)
