from main import events
import random


def get_price(season):
    if season == 'Summer':
        product_prices = {
            'Milk': 40,
            'Lettuce': 70,
            'Cheese': 330,
            'Eggs': 5,
            'Chicken': 130,
            'Tomatoes': 75,
            'Apples': 15,
            'Salmon': 138,
            'Pork': 160,
            'Potatoes': 12,
        }
        return product_prices
    elif season == 'Winter':
        product_prices = {
            'Milk': 45,
            'Lettuce': 90,
            'Eggs': 7,
            'Cheese': 350,
            'Chicken': 142,
            'Tomatoes': 90,
            'Apples': 22,
            'Salmon': 160,
            'Pork': 190,
            'Potatoes': 17,
        }
        return product_prices
    else:
        product_prices = {
            'Milk': 42,
            'Lettuce': 80,
            'Eggs': 6,
            'Cheese': 340,
            'Chicken': 138,
            'Tomatoes': 80,
            'Apples': 17,
            'Salmon': 145,
            'Pork': 175,
            'Potatoes': 15,
        }
        return product_prices


# Dictionary to map products to their categories
product_category = {
    'Milk': 'Dairy',
    'Eggs': 'Dairy',
    'Chicken': 'Meat',
    'Beef': 'Meat',
    'Tomatoes': 'Vegetables',
    'Apples': 'Fruits',
    'Fish': 'Seafood',
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
    'Fish': 3,       # fresh fish lasts about 3 days
    'Cheese': 90,    # hard cheese can last about 90 days
    'Lettuce': 5,    # lettuce lasts around 5 days
    'Beef': 7,       # beef can last up to a week
    'Potatoes': 90   # potatoes can be stored for several months
}


# Function to get the correct category for each product
def get_category(product):
    return product_category.get(product, 'Other')  # Returns 'Other' if product not found


def get_shelf_life(product):
    return shelf_life_dict[product]


def get_adjusted_quantity(product, quantity):
    if product == 'Eggs':
        return quantity * 10  # Assume eggs are sold by 10 pieces (adjust quantity accordingly)
    return quantity  # For kg and liters, the quantity remains the same


# Weights: 40% for 'None', 60% distributed across other events
event_weights = [0.4, 0.12, 0.12, 0.12, 0.12, 0.12]


# Choose an event based on the specified probabilities
def get_event():
    return random.choices(events, weights=event_weights, k=1)[0]
