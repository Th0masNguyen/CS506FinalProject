import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates


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
    max_value = int(x['WeeksUntilNextSale'].max()) + 1 # Include the max value in the bins
    
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
    
def WeeksUntilLinePlot(data, item):
    # Filter the data for the specified product
    product_data = data[data['Name'] == item]

    if product_data.empty:
        print(f"No data found for product: {item}")
        return

    # Plot the line graph for the specified product
    plt.figure(figsize=(10, 6))
    plt.plot(
        product_data["Week"], 
        product_data["WeeksUntilNextSale"], 
        marker='o', 
        label=item
    )
    
    plt.title(f"Weeks Until Next Sale for {item}")
    plt.xlabel("Week")
    plt.ylabel("Weeks Until Next Sale")
    plt.legend(title="Product")

    # Format the x-axis to display full dates (day, month, year)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    plt.xticks(rotation=45)  # Rotate the ticks for better readability
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

df = AllDf()
WeeksUntilLinePlot(df, 'chicken breasts boneless')
# WeeksUntilHisto(df, 'chicken breasts boneless')
