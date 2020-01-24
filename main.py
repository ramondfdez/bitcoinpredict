# Libraries
import pandas as pd

# Reading data from csv
bc = pd.read_csv('BTC-USD.csv')

# Converting to datetime
bc['Date'] = pd.to_datetime(bc.Date)

# Setting the index as the dates
bc.set_index('Date', inplace=True)

# Selecting only the dates from 2017-01-01 onwards
bc = bc[['Close']].loc['2017-01-01':]