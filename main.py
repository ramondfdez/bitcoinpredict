# Library Imports
import Tkinter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

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

#Prediction part

# Split the data into 80% training and 20% testing

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Set prediction_days_array equal to the last 30 rows of the original data set from the price column
prediction_days_array = np.array(df.drop(['Prediction'],1))[-prediction_days:]

# Create and train the Support Vector Machine (Regression) using the radial basis function
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.00001) #typical gamma
svr_rbf.fit(x_train, y_train) # fitting the model

# Testing Model: Score returns the accuracy of the prediction. 
# The best possible score is 1.0
svr_rbf_confidence = svr_rbf.score(x_test, y_test)
print("Accuracy: ", svr_rbf_confidence)

# Print the predicted value
svm_prediction = svr_rbf.predict(x_test)

# Print the model predictions for the next 'n=30' days
svm_prediction = svr_rbf.predict(prediction_days_array)

#Print the actual price for the next 'n' days, n=prediction_days=30 
df.dropna(inplace=True)

df.plot(y='Price',kind='line')
df.plot(y='Prediction',kind='line')

plt.show()