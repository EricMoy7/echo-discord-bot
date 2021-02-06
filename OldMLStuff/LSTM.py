import pandas as pd
import requests
from Config import *
import json
import matplotlib.pyplot as plt 
import numpy as np

import sklearn
from sklearn.preprocessing import MinMaxScaler

import os

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout

ticker = 'AAPL'
params = {
    "apikey": API_KEY,
    "periodType": "day",
    "period": 1,
    "frequencyType": "minute",
    "frequency": 5,
    "needExtendedHoursData": True
}
data = requests.get(f'https://api.tdameritrade.com/v1/marketdata/{ticker}/pricehistory', params=params)

#Convert JSON into pandas dataframe
json_data = json.loads(data.text)
candle_data = json_data['candles']
df = pd.DataFrame.from_records(candle_data)

total_rows = df["datetime"].count()
split_int = (total_rows*.8).round()

train = df.loc[:split_int]
test = df.loc[split_int:]

# Plotting Code
# train_x = train["datetime"]
# train_y = train["close"]
# test_x = test["datetime"]
# test_y = test["close"]
# plt.plot(train_x, train_y)
# plt.plot(test_x, test_y, color="red")
# plt.show()


# test.set_index("datetime", drop=True, inplace=True)

scaler = MinMaxScaler()
train_norm = scaler.fit_transform(train[['open','high','low','close','volume']])
train_norm = pd.DataFrame(train_norm, columns=['open','high','low','close','volume'])
# # train.set_index("datetime", drop=True, inplace=True)
# train.drop(labels=['datetime'])

print(train_norm.head())

def build_model(inputs, output_size, neurons, activ_func="linear",
                dropout=0.25, loss="mae", optimizer="adam"):
    model = Sequential()

    model.add(LSTM(neurons))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model

np.random.seed(202)
appl_model = build_model(train_norm, output_size = 1, neurons=20)
appl_hist = appl_model.fit(train, epochs=50, batch_size=1, verbose=2, shuffle=True)