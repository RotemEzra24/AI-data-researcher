import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Define realistic car profiles for the Israeli market
car_profiles = [
    {'Manufacturer': 'Toyota', 'Model': 'Corolla', 'Fuel_Type': 'Hybrid', 'Base_Price': 155000, 'Depreciation_Rate': 0.08},
    {'Manufacturer': 'Hyundai', 'Model': 'Tucson', 'Fuel_Type': 'Petrol', 'Base_Price': 175000, 'Depreciation_Rate': 0.11},
    {'Manufacturer': 'Kia', 'Model': 'Sportage', 'Fuel_Type': 'Petrol', 'Base_Price': 175000, 'Depreciation_Rate': 0.11},
    {'Manufacturer': 'BYD', 'Model': 'Atto 3', 'Fuel_Type': 'Electric', 'Base_Price': 165000, 'Depreciation_Rate': 0.13},
    {'Manufacturer': 'Tesla', 'Model': 'Model 3', 'Fuel_Type': 'Electric', 'Base_Price': 200000, 'Depreciation_Rate': 0.12},
    {'Manufacturer': 'Skoda', 'Model': 'Octavia', 'Fuel_Type': 'Petrol', 'Base_Price': 160000, 'Depreciation_Rate': 0.10},
    {'Manufacturer': 'Mazda', 'Model': 'Mazda 3', 'Fuel_Type': 'Petrol', 'Base_Price': 150000, 'Depreciation_Rate': 0.09},
    {'Manufacturer': 'Renault', 'Model': 'Megane', 'Fuel_Type': 'Diesel', 'Base_Price': 140000, 'Depreciation_Rate': 0.15},
    {'Manufacturer': 'Geely', 'Model': 'Geometry C', 'Fuel_Type': 'Electric', 'Base_Price': 155000, 'Depreciation_Rate': 0.13}
]

current_year = 2024
num_records = 3000
data = []

for _ in range(num_records):
    # Select a random car model profile
    profile = np.random.choice(car_profiles)
    
    # Generate realistic age and year
    year = np.random.randint(2018, current_year + 1)
    age = current_year - year
    
    # Calculate Mileage (avg 18,000 km per year + random noise)
    base_mileage = age * 18000
    mileage = int(max(1000, np.random.normal(base_mileage, 5000)))
    
    # Determine Hand (Ownership history count)
    if age == 0:
        hand = 1
    else:
        hand = np.random.randint(1, min(4, age + 2))
        
    # Determine Ownership Type (Leasing reduces value)
    ownership = np.random.choice(['Private', 'Leasing', 'Company'], p=[0.7, 0.2, 0.1])
    
    # Calculate Current Selling Price
    # Base depreciation formula based on age
    depreciation_factor = (1 - profile['Depreciation_Rate']) ** age
    
    # Penalties for condition
    mileage_penalty = (mileage - (age * 18000)) * 0.00005 # Small penalty per extra km
    hand_penalty = (hand - 1) * 0.02 # 2% drop per additional hand
    ownership_penalty = 0.12 if ownership != 'Private' else 0.0 # 12% drop for non-private
    
    # Calculate final price with some random market noise (+/- 4%)
    market_noise = np.random.uniform(0.96, 1.04)
    
    current_price = profile['Base_Price'] * depreciation_factor
    current_price = current_price * (1 - hand_penalty - ownership_penalty)
    current_price = current_price - (profile['Base_Price'] * mileage_penalty)
    current_price = current_price * market_noise
    
    # Ensure price boundaries
    current_price = min(profile['Base_Price'], max(20000, current_price))
    
    data.append({
        'Manufacturer': profile['Manufacturer'],
        'Model': profile['Model'],
        'Year': year,
        'Fuel_Type': profile['Fuel_Type'],
        'Original_Price_ILS': int(profile['Base_Price']),
        'Current_Price_ILS': int(current_price),
        'Mileage_km': mileage,
        'Hand': hand,
        'Ownership_Type': ownership
    })

# Save to CSV
df = pd.DataFrame(data)
df.to_csv('israel_car_market_prices.csv', index=False)
print("Dataset 'israel_car_market_prices.csv' generated successfully.")