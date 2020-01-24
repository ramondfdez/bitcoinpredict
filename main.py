# Library Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Loading/Reading in the Data
df = pd.read_csv("BTC-USD.csv")

#Remove innecesary columns
df.drop(['Date'], 1, inplace=True)
df.drop(['High'], 1, inplace=True)
df.drop(['Low'], 1, inplace=True)
df.drop(['Adj Close'], 1, inplace=True)
df.drop(['Volume'], 1, inplace=True)


#A variable for predicting 'n' days out into the future
prediction_days = 30 #n = 30 days

#Create Price Column

df['Price'] = (df['Open'] + df['Close'])/2

#Remove temporal columns
df.drop(['Open'], 1, inplace=True)
df.drop(['Close'], 1, inplace=True)

#Create another column for predictions, we need to shift it up
df['Prediction'] = df[['Price']].shift(-prediction_days)

#Converting it to numpy array 

# Convert the dataframe to a numpy array and drop the prediction column
X = np.array(df.drop(['Prediction'],1))

#Remove the last 'n' rows where 'n' is the prediction_days
X= X[:len(df)-prediction_days]

# Convert the dataframe to a numpy array (All of the values including the NaN's)
y = np.array(df['Prediction'])

# Get all of the y values except the last 'n' rows
y = y[:-prediction_days]

