import keras
import np as np
import pandas as pd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import optimizers
from yahoo_fin import stock_info as si
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
"""
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 6} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)
"""



def output():
    scaler = MinMaxScaler(feature_range=(0, 1))

    start = input("What is the start date?  DD/MM/YYYY ")
    end = input("What is the end date? DD/MM/YYYY ")
    time = input("How far forward do you want the model to predict?, increment between 1-5 ")
    ticker = input("What stock do you want to predict, ticker name (Caps) ")


    try:
        si.get_live_price(ticker)
    except BaseException:
        raise


    week = si.get_data(ticker, start_date=start, end_date=end)

    week = week.iloc[:, 0]
    week.to_numpy()
    stock_price = []

    for i in range(0, week.shape[0]):
        stock_price.append(week[i])
    stock_price = [stock_price]

    stock_price = np.asarray(stock_price, dtype=np.float32)

    stock_price = np.reshape(stock_price, (stock_price.shape[1], stock_price.shape[0]))

    training_processed = stock_price

    #training,testing = train_test_split(training_processed, train_size= 0.7, test_size= 0.3, shuffle= False)

    training = training_processed
    testing = training_processed
    training_scaled = scaler.fit_transform(training)
    testing_scaled = scaler.fit_transform(testing)

    features_set = []
    labels = []

    for i in range(len(training_scaled)):
        features_set.append(training_scaled[i])
    features_set.remove(features_set[i])

    for i in range(1, len(training_scaled)):
        labels.append(training_scaled[i])

    features_set, labels = np.array(features_set), np.array(labels)

    features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=100, return_sequences=True, activation= "relu"))
    model.add(Dropout(0.2))

    model.add(LSTM(units=100, return_sequences=True, activation= "relu"))
    model.add(Dropout(0.2))

    model.add(LSTM(units=100, return_sequences=True, activation= "relu"))
    model.add(Dropout(0.2))

    model.add(LSTM(units=100, return_sequences=True, activation= "relu"))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(features_set, labels, epochs=50, batch_size=64)

    a = []
    counter = 0
    a.append(np.reshape(training_scaled[training_scaled.size - 1], (1, 1, 1)))
    while counter < int(time):
        a.append(model.predict(np.reshape(a[len(a) - 1], (1, 1, 1))))
        counter += 1



    a = np.reshape(a, (len(a), 1))

    temp = np.reshape(testing_scaled, (testing_scaled.size, 1, 1))
    temp = model.predict(temp)
    temp = np.reshape(temp, (testing_scaled.size, 1))
    temp = scaler.inverse_transform(temp)
    a = scaler.inverse_transform(a)

    a = np.append(temp,a)
    a = a.tolist()
    training_scaled = scaler.inverse_transform(training_scaled)
    training_scaled = training_scaled.tolist()
    """
    plt.figure(figsize=(10, 6))
    plt.plot(training_scaled, color='blue', label='Actual ' + ticker + ' Stock Price')
    plt.plot(a, color='red', label='Predicted ' + ticker + ' Stock Price')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
    """
    return a,training_scaled




output()


