import numpy as np
import pandas as pd

# Define the number of profiles to generate
num_profiles = 5000

# Generate random ages between 18 and 70
ages = np.random.randint(18, 71, size=num_profiles)

# Generate random genders (0 for male, 1 for female)
genders = np.random.randint(0, 2, size=num_profiles)

# Generate random interests (5 binary features)
interests = np.random.randint(0, 2, size=(num_profiles, 5))

# Generate random average ratings between 1 and 5
average_ratings = np.random.uniform(1, 5, size=num_profiles)

# Combine all features into a single dataset
data = np.column_stack((ages, genders, interests, average_ratings))

# Create a DataFrame
columns = ['age', 'gender', 'interest_sports', 'interest_music', 'interest_technology', 'interest_travel', 'interest_food', 'average_rating']
df = pd.DataFrame(data, columns=columns)

# Save the dataset to a CSV file
df.to_csv('synthetic_user_profiles.csv', index=False)

print(f"Generated dataset with {num_profiles} user profiles and saved to 'synthetic_user_profiles.csv'")
