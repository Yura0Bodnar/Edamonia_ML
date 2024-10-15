import random
import pandas as pd
from additional_functions import *
from datetime import datetime, timedelta

import random
from datetime import timedelta


def get_price(season, product):
    if season == 'Summer':
        product_prices = {
            'Milk': 40,
            'Lettuce': 70,
            'Cheese': 330,
            'Eggs': 50,
            'Chicken': 130,
            'Tomatoes': 75,
            'Apples': 15,
            'Salmon': 138,
            'Pork': 160,
            'Potatoes': 12,
        }
    elif season == 'Winter':
        product_prices = {
            'Milk': 45,
            'Lettuce': 90,
            'Eggs': 70,
            'Cheese': 350,
            'Chicken': 142,
            'Tomatoes': 90,
            'Apples': 22,
            'Salmon': 160,
            'Pork': 190,
            'Potatoes': 17,
        }
    else:  # For Spring and Autumn
        product_prices = {
            'Milk': 42,
            'Lettuce': 80,
            'Eggs': 60,
            'Cheese': 340,
            'Chicken': 138,
            'Tomatoes': 80,
            'Apples': 17,
            'Salmon': 145,
            'Pork': 175,
            'Potatoes': 15,
        }

    # Return the price for the specific product
    return product_prices.get(product, None)  # None if product not found


# Dictionary to map products to their categories
product_category = {
    'Milk': 'Dairy',
    'Eggs': 'Dairy',
    'Chicken': 'Meat',
    'Pork': 'Meat',
    'Tomatoes': 'Vegetables',
    'Apples': 'Fruits',
    'Salmon': 'Seafood',
    'Cheese': 'Dairy',
    'Lettuce': 'Vegetables',
    'Potatoes': 'Vegetables'
}

# Shelf life dictionary (in days)
shelf_life_dict = {
    'Milk': 7,       # milk can last around 7 days
    'Eggs': 21,      # eggs can last about 3 weeks
    'Chicken': 5,    # chicken can be stored for up to 5 days
    'Tomatoes': 10,  # tomatoes last about 10 days
    'Apples': 30,    # apples can last for a month
    'Salmon': 3,       # fresh fish lasts about 3 days
    'Cheese': 90,    # hard cheese can last about 90 days
    'Lettuce': 5,    # lettuce lasts around 5 days
    'Pork': 7,       # beef can last up to a week
    'Potatoes': 90   # potatoes can be stored for several months
}


# Function to get the correct category for each product
def get_category(product):
    return product_category.get(product, 'Other')  # Returns 'Other' if product not found


def get_shelf_life(product):
    return shelf_life_dict[product]


# Weights: 40% for 'None', 60% distributed across other events
event_weights = [0.4, 0.12, 0.12, 0.12, 0.12, 0.12]


# Choose an event based on the specified probabilities
def get_event():
    return random.choices(events, weights=event_weights, k=1)[0]


def get_stock(num_customers):
    if num_customers > 70:
        stock_left = random.randint(20, 50)  # Large stock if few visitors
        return stock_left
    else:
        stock_left = random.randint(0, 20)  # Small stock if more visitors
        return stock_left


def get_average_check():
    average_check_per_customer = random.uniform(200, 600)
    return average_check_per_customer


def determine_season(date):
    month = date.month
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'


# Weather conditions based on the season
seasonal_weather = {
    'Winter': ['Snowy', 'Cold', 'Cloudy'],
    'Spring': ['Sunny', 'Rainy', 'Cloudy'],
    'Summer': ['Sunny', 'Hot', 'Cloudy'],
    'Autumn': ['Rainy', 'Cloudy', 'Windy']
}


def next_purchase(stock_left, product):
    if stock_left < 10:
        days_until_next_purchase = 1  # Restock immediately if stock is low
        return days_until_next_purchase
    else:
        days_until_next_purchase = get_shelf_life(product)  # Otherwise, use shelf life
        return days_until_next_purchase


# New logic to determine purchase quantity
def determine_quantity(num_customers, stocks, season, product):
    base_quantity = 50  # Base quantity for all products

    # Increase quantity if many customers
    if num_customers > 300:
        base_quantity += 20  # More customers, larger purchase

    # Increase quantity if low stock
    if stocks < 10:
        base_quantity += 30  # Urgent need to restock

    # Seasonal adjustments (e.g., more vegetables in summer)
    if season == 'Summer' and product in ['Tomatoes', 'Lettuce']:
        base_quantity += 15  # Summer demand for fresh produce

    # Adjust for product type (e.g., more demand for milk in winter)
    if season == 'Winter' and product == 'Milk':
        base_quantity += 10

    return base_quantity


# Function to generate number of customers based on various factors
def generate_num_customers(season, weather, event, day_of_week):
    base_customers = 250  # Base number of visitors in average conditions

    # Adjust for season
    if season == 'Summer':
        base_customers += 50  # More visitors in summer
    elif season == 'Winter':
        base_customers -= 30  # Fewer visitors in winter

    # Adjust for weather
    if weather == 'Sunny':
        base_customers += 30  # More visitors in sunny weather
    elif weather in ['Rainy', 'Snowy', 'Stormy']:
        base_customers -= 40  # Fewer visitors in bad weather

    # Adjust for events
    if event != 'None':
        base_customers += 100  # Significant increase if there's an event

    # Adjust for day of the week (weekends typically attract more customers)
    if day_of_week in ['Saturday', 'Sunday']:
        base_customers += 50  # More visitors on weekends

    # Generate final number of customers with some randomness
    num_customers = random.randint(base_customers - 50, base_customers + 50)

    # Ensure number of customers stays within realistic bounds
    num_customers = max(50, min(500, num_customers))

    return num_customers


# Parameters for data generation
products = ['Milk', 'Eggs', 'Chicken', 'Tomatoes', 'Apples', 'Salmon', 'Cheese', 'Lettuce', 'Pork', 'Potatoes']
seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weather_conditions = ['Sunny', 'Rainy', 'Snowy', 'Cloudy', 'Stormy', 'Hot', 'Cold']
events = ['None', 'Holiday Special', 'Discount', 'Corporate Event', 'Special Promotion', 'Seasonal Event']


# Helper function to generate sequential dates with a small range between days
# Helper function to generate sequential dates with a maximum gap of 7 days for each product
def generate_sequential_date(previous_date, product_purchase_tracker, product, min_days=0, max_days=3):
    """
    Generate a sequential date for a product, ensuring that it is purchased at least once a week.
    The function tracks the last purchase date for each product and ensures a purchase every 7 days.
    """
    last_purchase_date = product_purchase_tracker.get(product, previous_date)

    # Add a random number of days between min_days and max_days to ensure purchases are spaced out
    delta_days = random.randint(min_days, max_days)

    # Calculate new purchase date
    new_date = last_purchase_date + timedelta(days=delta_days)

    # Ensure the new date is not more than 4 days from the last purchase
    if (new_date - last_purchase_date).days > 4:
        new_date = last_purchase_date + timedelta(days=4)

    # Update the tracker with the new purchase date for the product
    product_purchase_tracker[product] = new_date

    return new_date


# Function to generate sequential dates with an emphasis on peak seasons
def generate_date(previous_date, product, product_purchase_tracker, min_days=0, max_days=3, peak_seasons=None):
    """
    Generate the next purchase date for a product, ensuring it aligns with peak seasons and minimum weekly purchases.
    """
    # If peak seasons are not specified, treat all dates equally
    if peak_seasons is None:
        peak_seasons = ['Winter', 'Summer']  # Default peak seasons

    # Generate the next sequential date for the product
    new_date = generate_sequential_date(previous_date, product_purchase_tracker, product, min_days, max_days)

    # Check if the new date is in a peak season
    if determine_season(new_date) in peak_seasons:
        # Increase the likelihood of selecting dates within peak seasons
        if random.random() < 0.7:  # 70% chance to keep the date within a peak season
            return new_date

    # Return the new date even if it's not in a peak season
    return new_date


# Generate synthetic data
def generate_synthetic_data(n_rows):
    data = []

    # Date range for purchases
    current_date = datetime(2014, 1, 1)
    end_date = datetime(2024, 10, 1)
    # Track the last purchase date for each product to ensure weekly purchases
    product_purchase_tracker = {product: current_date for product in products}

    for _ in range(n_rows):
        product = random.choice(products)

        # Generate the next purchase date for this product
        current_date = generate_date(current_date, product, product_purchase_tracker)

        # Ensure the date doesn't exceed the end_date
        if current_date > end_date:
            break  # Stop generation if we've reached the end date

        season = determine_season(current_date)
        weather = random.choice(seasonal_weather[season])
        product = random.choice(products)
        event = get_event()
        day_of_week = days_of_week[current_date.weekday()]
        num_customers = generate_num_customers(season, weather, event, day_of_week)
        stocks = get_stock(num_customers)
        days_until_next_purchase = next_purchase(stocks, product)
        category = get_category(product)
        quantity = determine_quantity(num_customers, stocks, season, product)
        unit_price = get_price(season, product)
        sales = round(num_customers * get_average_check(), 2)
        shelf_life = get_shelf_life(product)

        # Append row data
        data.append([current_date, day_of_week, season, weather, product, category, unit_price,
                     num_customers, sales, stocks, shelf_life, days_until_next_purchase,  event, quantity])

    # Create DataFrame
    columns = ['Date', 'Day_of_Week', 'Season', 'Weather', 'Product', 'Category', 'Unit_Price',
               'Num_Customers', 'Sales', 'Stocks', 'Shelf_Life', 'Days_Until_Next_Purchase', 'Event', 'Purchase_Quantity']

    df = pd.DataFrame(data, columns=columns)
    return df


# Generate synthetic data
synthetic_data = generate_synthetic_data(100000)

# To display the first 5 rows of the DataFrame
print(synthetic_data.head())

# Or, if you'd like to export it to a CSV file
synthetic_data.to_csv('synthetic_purchase_data.csv', index=False)
