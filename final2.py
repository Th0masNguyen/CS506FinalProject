import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Load data
df = pd.read_csv("data.csv")

# Convert Week to datetime
df['Week'] = pd.to_datetime(df['Week'])

# Sort by Name and Week
df = df.sort_values(by=['Name', 'Week'])

# Calculate days until next sale
df['NextSaleDate'] = df.groupby('Name')['Week'].shift(-1)
df['DaysUntilNextSale'] = (df['NextSaleDate'] - df['Week']).dt.days

# Feature engineering
df['OnSale'] = df['RegPrice'] > df['SalePrice']

# Drop rows with NaN in target
df = df.dropna(subset=['DaysUntilNextSale'])

# Encode categorical columns
df = pd.get_dummies(df, columns=['Category'], drop_first=True)

# Define features and target
X = df.drop(columns=['Week', 'NextSaleDate', 'DaysUntilNextSale', 'Name'])
y = df['DaysUntilNextSale']

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))

# Predict for ground beef
ground_beef = df[df['Name'] == 'ground beef'].drop(columns=['Week', 'NextSaleDate', 'DaysUntilNextSale', 'Name'])
next_sale = model.predict(ground_beef)
print("Predicted days until next sale for ground beef:", next_sale)
print(X)
print(df)
