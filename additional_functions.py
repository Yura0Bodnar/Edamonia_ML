def get_price(season):
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
        return product_prices
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
        return product_prices
    else:
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
        return product_prices


# Dictionary to map products to their categories
product_category_map = {
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


# Function to get the correct category for each product
def get_category(product):
    return product_category_map.get(product, 'Other')  # Returns 'Other' if product not found
