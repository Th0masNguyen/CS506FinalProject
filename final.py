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


df = AllDf()
WeeksUntilHisto(df, 'chicken breasts boneless')
