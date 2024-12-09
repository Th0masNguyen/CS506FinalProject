import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def ProcessData():
  # Load the data from CSV
  data = pd.read_csv('data.csv')

  # Convert the Week column to datetime format
  data['Week'] = pd.to_datetime(data['Week'])

  # Add a new column to determine if the item is on sale
  data['OnSale'] = data['RegPrice'] > data['SalePrice']

  # Pivot the data to create the desired structure
  pivot_df = data.pivot_table(
      index='Week',  # Rows are weeks
      columns='Name',  # Columns are item names
      values='OnSale',  # Values are the 'OnSale' boolean
      aggfunc='first'  # Ensure unique values are retained
  )

  # Fill NaN values with False (indicating the item is not on sale)
  pivot_df = pivot_df.fillna(False)

  # Sort rows by the Week column
  pivot_df = pivot_df.sort_index()

  # Ensure the column names are user-friendly
  pivot_df.columns.name = None
  
  return pivot_df

def add_weeks_until_next_sale(df, item_name):
    # Ensure the index is a datetime index for proper calculations
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    
    # Extract the sale information for the given item
    on_sale = df[item_name].astype(bool)
    
    # Initialize a new column with NaN values
    weeks_until_next_sale = pd.Series(np.nan, index=df.index)
    
    # Iterate through the DataFrame to calculate weeks until next sale
    for i in range(len(on_sale)):
        if on_sale.iloc[i]:  # If the item is on sale, set 0
            weeks_until_next_sale.iloc[i] = 0
        else:
            # Find the index of the next True value (next sale)
            future_sales = on_sale.iloc[i + 1:]
            if future_sales.any():  # Check if there are any future sales
                next_sale_idx = future_sales.idxmax()
                weeks_until_next_sale.iloc[i] = (next_sale_idx - on_sale.index[i]).days // 7
            else:
                weeks_until_next_sale.iloc[i] = None  # No future sale available

    # Add the new column to the DataFrame
    df['weeksUntilNextItemSale'] = weeks_until_next_sale
    
    return df

def add_weeks_since_last_sale(df, item_name):
    # Ensure the index is a datetime index for proper calculations
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    
    # Extract the sale information for the given item
    on_sale = df[item_name].astype(bool)
    
    # Initialize a new column with NaN values
    weeks_since_last_sale = pd.Series(np.nan, index=df.index)
    
    # Iterate through the DataFrame to calculate weeks since the last sale
    for i in range(len(on_sale)):
        if on_sale.iloc[i]:  # If the item is on sale, set 0
            weeks_since_last_sale.iloc[i] = 0
        else:
            # Find the index of the last True value (last sale)
            past_sales = on_sale.iloc[:i]
            if past_sales.any():  # Check if there are any past sales
                last_sale_idx = past_sales[::-1].idxmax()  # Reverse to get the most recent
                weeks_since_last_sale.iloc[i] = (on_sale.index[i] - last_sale_idx).days // 7
            else:
                weeks_since_last_sale.iloc[i] = None  # No past sale available

    # Add the new column to the DataFrame
    df['weeksSinceLastItemSale'] = weeks_since_last_sale
    
    return df
  
def train_weeks_until_sale_model(df, item_name):
    # Add weeksSinceLastItemSale and weeksUntilNextItemSale columns
    df = add_weeks_since_last_sale(df, item_name)
    df = add_weeks_until_next_sale(df, item_name)
    
    updated_df = df.copy()
    
    # Drop rows where weeksUntilNextItemSale is NaN (unpredictable rows)
    df = df.dropna(subset=['weeksUntilNextItemSale'])
    df = df.dropna(subset=['weeksSinceLastItemSale'])
    
    # Drop rows with sale weeks for item name
    # df = df[df['weeksSinceLastItemSale'] != 0]
    
    # Select features and target
    X = df.drop(columns=['weeksUntilNextItemSale'])
    y = df['weeksUntilNextItemSale']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a model (Random Forest Regressor as an example)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    
    return model, updated_df


data = ProcessData()
# Train the model for "itemName"
itemName = 'ground beef'
model, updated_df = train_weeks_until_sale_model(data, itemName)

# Example prediction
X = updated_df.drop(columns=['weeksUntilNextItemSale'])
lastWeek = X.iloc[-1].to_frame().T  # Convert last row to DataFrame
lastWeek['weeksSinceLastItemSale'] = 1
print(lastWeek)


# Predict weeks until next sale
predicted_weeks = model.predict(lastWeek)
print(f"Predicted weeks until next sale: {predicted_weeks[0]:.2f}")