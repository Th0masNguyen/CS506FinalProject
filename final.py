import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_squared_error


def AllDf():
  file_path = 'data.csv'
  data = pd.read_csv(file_path)
  data = data[data['Category'] == 'Meat']
  data['Week'] = pd.to_datetime(data['Week'])
  # data['Month'] = data['Week'].dt.month
  # data['Discount'] = (data['RegPrice'] - data['SalePrice']) / data['RegPrice']
  data = data.sort_values(by='Week', ascending=True)

  # data['PrevSalePrice'] = data.groupby('Name')['SalePrice'].shift(1)
  data['WeeksSinceLastSale'] = data.groupby('Name')['Week'].diff().dt.days / 7
  data['WeeksUntilNextSale'] = data.groupby('Name')['WeeksSinceLastSale'].shift(-1)
  data = data.dropna()
  selectedCols = ['Week', 'Name', 'WeeksSinceLastSale', 'WeeksUntilNextSale']
  data = data[selectedCols]
  return data

def WeeksSinceUntilRelation(data):
  # Scatter plot to visualize the relationship
  plt.figure(figsize=(10, 6))
  plt.scatter(data['WeeksSinceLastSale'], data['WeeksUntilNextSale'], alpha=0.5)
  plt.xlabel('Weeks Since Last Sale')
  plt.ylabel('Weeks Until Next Sale')
  plt.title('Relationship Between Weeks Since Last Sale and Weeks Until Next Sale')
  plt.grid(True)

  # Show the plot
  plt.tight_layout()
  plt.show()

def WeeksSinceHisto(data, item):
    x = data[data['Name'] == item]
    min_value = int(x['WeeksSinceLastSale'].min())
    max_value = int(x['WeeksSinceLastSale'].max()) + 1  # Include the max value in the bins
    
    # Calculate the mean of WeeksSinceLastSale
    mean_value = x['WeeksSinceLastSale'].mean()
    
    # Create bins with an interval of 1
    bins = range(min_value, max_value + 1)
    
    # Plot histogram of WeeksSinceLastSale
    plt.figure(figsize=(10, 6))
    plt.hist(x['WeeksSinceLastSale'], bins=bins, edgecolor='black')
    plt.xlabel('Weeks Since Last Sale')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Weeks Since Last Sale for {item}')
    plt.grid(axis='y')
    
    # Add a vertical line for the mean
    plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=1)
    plt.text(mean_value, plt.gca().get_ylim()[1] * 0.9, f'Mean: {mean_value:.2f}', color='red')
    
    # Show the plot
    plt.tight_layout()
    plt.show()

def WeeksUntilHisto(data, item):
    x = data[data['Name'] == item]
    min_value = int(x['WeeksUntilNextSale'].min())
    max_value = int(x['WeeksUntilNextSale'].max()) + 1  # Include the max value in the bins
    
    # Calculate the mean of WeeksUntilNextSale
    mean_value = x['WeeksUntilNextSale'].mean()
    
    # Create bins with an interval of 1
    bins = range(min_value, max_value + 1)
    plt.figure(figsize=(10, 6))
    plt.hist(x['WeeksUntilNextSale'], bins=bins, edgecolor='black')
    plt.xlabel('Weeks Until Next Sale')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Weeks Until Next Sale for {item}')
    plt.grid(axis='y')
    
    # Add a vertical line for the mean
    plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=1)
    plt.text(mean_value, plt.gca().get_ylim()[1] * 0.9, f'Mean: {mean_value:.2f}', color='red')
    
    # Show the plot
    plt.tight_layout()
    plt.show()
  
def predict_weeks_until_next_sale(df):
    # Drop rows where WeeksUntilNextSale is None
    df_clean = df.dropna(subset=['WeeksUntilNextSale'])

    # Prepare data for training
    X = np.array(range(len(df_clean))).reshape(-1, 1)  # Create index as feature
    y = df_clean['WeeksUntilNextSale'].values

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict the number of weeks until the next sale (for the last week in the dataset)
    prediction = model.predict(np.array([[len(df_clean)]]))  # Predict next week

    return prediction[0]

def SingleItemDf(csv_file_path, item_name):
    # Read the CSV file into a DataFrame, ensuring the 'Week' column is parsed as dates
    df = pd.read_csv(csv_file_path, parse_dates=['Week'])
    
    # Sort DataFrame by 'Week' to ensure the rows are ordered chronologically
    df = df.sort_values('Week')
    
    # Create a boolean column for item sales (True if the item is on sale in that row)
    df['OnSale'] = df['Name'] == item_name

    # Create a DataFrame to hold the results
    unique_weeks = df['Week'].unique()
    result_df = pd.DataFrame(unique_weeks, columns=['Week'])

    
    # Mark whether the item was on sale for each week
    result_df['OnSale'] = result_df['Week'].isin(df.loc[df['OnSale'], 'Week'])
   

     # Calculate the weeks since the last sale for each week
    last_sale = None
    weeks_since_last_sale = []
    
    for week in result_df['Week']:
        if result_df[result_df['Week'] == week]['OnSale'].values[0]:  # Sale happened in this week
            last_sale = week
            weeks_since_last_sale.append(0)
        else:
            # Calculate weeks since the last sale, if a sale happened before this week
            if last_sale is not None:
                weeks_since_last_sale.append((week - last_sale).days // 7)
            else:
                weeks_since_last_sale.append(None)  # No sales yet, so weeks_since_last_sale is None

    result_df['WeeksSinceLastSale'] = weeks_since_last_sale
    
    next_sale_index = None
    for i in range(len(result_df) - 1, -1, -1):
        if result_df.loc[i, 'OnSale']:
            next_sale_index = i  # Update next sale index
            result_df.loc[i, 'WeeksUntilNextSale'] = 0
        else:
            if next_sale_index is not None:
                result_df.loc[i, 'WeeksUntilNextSale'] = next_sale_index - i
            else:
                result_df.loc[i, 'WeeksUntilNextSale'] = None

    return result_df
  
  
def Predict(df):
  result_df = df

  # Drop rows with missing values for WeeksUntilNextSale
  result_df = result_df.dropna(subset=['WeeksSinceLastSale', 'WeeksUntilNextSale'])

  # Define X and y
  X = result_df[['WeeksSinceLastSale']]
  y = result_df['WeeksUntilNextSale']

  # Split data into training and test sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

  # Create and train the linear regression model
  model = LinearRegression()
  model.fit(X_train, y_train)

  # Predict on the test set
  y_pred = model.predict(X_test)

  # Evaluate the model
#   mse = mean_squared_error(y_test, y_pred)
#   print(f'Mean Squared Error: {mse}')

  # Automatically make a prediction based on the last row's WeeksSinceLastSale
  last_weeks_since_last_sale = result_df.iloc[-1]['WeeksSinceLastSale']
  predicted_weeks_until_next_sale = model.predict([[last_weeks_since_last_sale]])
  print(f'Predicted Weeks Until Next Sale (based on last row): {predicted_weeks_until_next_sale[0]}')
  
  test1 = 4
  test2 = model.predict([[test1]])
  print(test2)

df = SingleItemDf('data.csv', 'chicken breasts boneless')
Predict(df)
# df = AllDf()
# WeeksUntilHisto(df, 'chicken breasts boneless')
