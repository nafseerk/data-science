import pandas as pd #for data handling
import quandl #for data set 
import math
import datetime
import numpy as np

# For learning algorithms
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression  

# For plotting
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

quandl.ApiConfig.api_key = "Your API key here"
df = quandl.get('WIKI/GOOGL')

# Data cleaning and feature extraction
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0 
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0 
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
df.fillna(-9999, inplace=True) #replace missing data with a number(which doesn't affect our predication much. It will be treated as an outlier)

# the target output - the closing price in future
forecast_col = 'Adj. Close'
forecast_out = int(math.ceil(0.01 * len(df))) #No of days into future we are going to predict the stock close price
df['label'] = df[forecast_col].shift(-forecast_out) # Note: we are shifting up Adj. close by forecast_out times. So this will create some NaNs at end which has to be be taken care of

#preprocessing
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X) # when adding additional data in future make sure to scale them alongwith training data
X = X[:-forecast_out]
X_future = X[-forecast_out:] # this is the last rows for which we are going to predicat the closing price
df.dropna(inplace=True)
y = np.array(df['label'])  # convention to use X for features and y for label

#take test data from training set for validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# training with LinearRegression and testing the accuracy
clf = LinearRegression()
clf.fit(X_train, y_train) # fit is training 
accuracy = clf.score(X_test, y_test) # score is testing
#print(accuracy)

# training with SVM and testing the accuracy
#clf = svm.SVR()
#clf.fit(X_train, y_train) 
#accuracy = clf.score(X_test, y_test)
#print(accuracy)

# make prediction for the labels i.e the closing price for the X_future set
forecast_set = clf.predict(X_future)  

#from df calculate the first nextdate for prediction
df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_date_timestamp = last_date.timestamp()
oneday = 86400
next_date_timestamp = last_date_timestamp + oneday

#fill new rows in df for the forecasted dates
for i in forecast_set: 
    nextdate = datetime.datetime.fromtimestamp(next_date_timestamp)
    next_date_timestamp += oneday
    df.loc[nextdate] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

#Plot
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


