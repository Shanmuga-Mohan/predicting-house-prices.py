import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
data = {
    'Bedrooms': np.random.randint(1, 6, size=50),
    'SquareFootage': np.random.randint(500, 3500, size=50),
    'Price': np.random.randint(100000, 500000, size=50)
}

# Create a DataFrame
df = pd.DataFrame(data)

# Display the first few rows of the dataset
print(df.head())

# Save the dataset to a CSV file (optional)
df.to_csv('house_prices_sample.csv', index=False)
import pandas as pd

# Load the dataset from a CSV file
df = pd.read_csv('house_prices_sample.csv')

# Display the first few rows to confirm it's loaded correctly
print(df.head())
# Assuming df is already defined as above

# Split the data into features (X) and target (y)
X = df[['Bedrooms', 'SquareFootage']]
y = df['Price']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
from sklearn.metrics import mean_squared_error, r2_score
y_pred = model.predict(X_test)

# Calculate and print evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")
