import pandas as pd
import requests
from Config import *
import json
import matplotlib.pyplot as plt 
import numpy as np

import sklearn
import sklearn.preprocessing as preprocessing
from collections import deque
import random
import time

import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

SEQ_LEN = 200
FUTURE_PERIOD_PREDICT = 5
EPOCHS = 10
BATCH_SIZE = 64
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"

def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

def preprocess_df(df):
    df = df.drop('future', 1)
    for col in df.columns:
        if col != 'target':
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)
    df.dropna(inplace=True)
    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)

    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])
    random.shuffle(sequential_data)

    buys = []
    sells = []

    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])
    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys), len(sells))

    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys + sells

    random.shuffle(sequential_data)

    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), y

ticker = 'AAPL'
params = {
    "apikey": API_KEY,
    "periodType": "day",
    "period": 5,
    "frequencyType": "minute",
    "frequency": 5,
    "needExtendedHoursData": True
}
data = requests.get(f'https://api.tdameritrade.com/v1/marketdata/{ticker}/pricehistory', params=params)

#Convert JSON into pandas dataframe
json_data = json.loads(data.text)
candle_data = json_data['candles']
df = pd.DataFrame.from_records(candle_data)

# total_rows = df["datetime"].count()
# split_int = (total_rows*.8).round()

# train = df.loc[:split_int]
# test = df.loc[split_int:]

df.set_index("datetime", inplace=True)
df = df[['close', 'volume']]

df['future'] = df["close"].shift(-FUTURE_PERIOD_PREDICT)

df['target'] = list(map(classify, df['close'], df['future']))

# print(df[['close', 'future', 'target']].head(10))

times = sorted(df.index.values)
last_5pct = times[-int(0.05*len(times))]

validation_df = df[(df.index >= last_5pct)]
main_df = df[(df.index < last_5pct)]

train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_df)
# preprocess_df(main_df)


print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")

train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
validation_x = np.asarray(validation_x)
validation_y = np.asarray(validation_y)

model = Sequential()
model.add(LSTM(128, input_shape=train_x.shape[1:], return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape=train_x.shape[1:], return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape=train_x.shape[1:]))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(2, activation="softmax"))

opt = tf.keras.optimizers.Adam(learning_rate=0.001, decay= 1e-6)

model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

filepath = "RNN_Final-something"  # unique file name that will include the epoch and the validation acc for that epoch
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) # saves only the best ones

history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, validation_y),
    callbacks=[tensorboard, checkpoint],
)