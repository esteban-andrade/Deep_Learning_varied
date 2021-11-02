import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


def DataPreparation():
    dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
    # we take the range from 1 to 2 to chooose the value that we want as we need a vector
    # we only take the first index
    training_set = dataset_train.iloc[:, 1:2].values

    """
    FEATURE SCALING

    2 ways for feature scaling are 
    * standardisation
    * Normalisation
    """
    # in this case we use normalization
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    """Creating Data Structure with time steps"""
    X_train = []  # input of NN
    y_train = []  # output of NN

    steps = 60
    for i in range(steps, 1258):
        # we append previous 60 stock prices
        X_train.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0])

    # We make them numpu arrays
    X_train, y_train = np.array(X_train), np.array(y_train)

    """RESHAPING"""
    # we need to add another dimension
    # X_train.shape[1] gives number of columns
    # X_train.shape[0] number of rows
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    return X_train, y_train, dataset_train


def Building_Training(data):

    X_train = data[0]
    y_train = data[1]
    print(len(X_train))
    # initialise RNN
    regressor = Sequential()  # we are predicting continous value

    # ADD LSTM Layer & Dropout regulation
    regressor.add(LSTM(units=50,
                       return_sequences=True,
                       input_shape=(X_train.shape[1], 1)))  # we only show the Y axis and the indicator observation as the other param will be already taken into account

    regressor.add(Dropout(0.2))  # regularization to avoid overfitting

    # add second layer
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))

    # 3rd layer
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))

    # 4rth layer
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))  # 20 % of neurons will be ignored

    # output later
    regressor.add(Dense(units=1))

    # compile
    # regressor.compile(optimizer=tf.keras.optimizers.Adam(),
    #                   loss=tf.keras.losses.MeanSquaredError())

    regressor.compile(optimizer='adam', loss='mean_squared_error')

    """FIT RNN To Training Set"""
    regressor.fit(X_train, y_train, epochs=100, batch_size=32)
    return regressor


def Predict(model, data):


    dataset_train = data[2]
    dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
    real_stock_price = dataset_test.iloc[:, 1:2].values

    # get stock prize at 2017
    # we will need training set and testset for eacxh day of 2017
    dataset_total = pd.concat(
        (dataset_train["Open"], dataset_test["Open"]), axis=0)  # axis 0 --> vertical axis

    # we need the prevoid 60 firts financial dates
    inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
    inputs = inputs.reshape(-1, 1)
    sc = MinMaxScaler(feature_range=(0, 1))
    sc.fit_transform(inputs)
    inputs = sc.transform(inputs)

    X_test = []
    for i in range(60, 80):
        X_test.append(inputs[i-60:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # predict
    predicted_strock_prize = model.predict(X_test)
    # We need to invert it to get the original value and not the scaled values
    predicted_strock_prize = sc.inverse_transform(predicted_strock_prize)


# Visualising the results
    plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
    plt.plot(predicted_strock_prize, color='blue',
             label='Predicted Google Stock Price')
    plt.title('Google Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Google Stock Price')
    plt.legend()
    plt.show()

    return predicted_strock_prize


def main():
    data = DataPreparation()
    rnn = Building_Training(data)
    Predict(rnn,data)

if __name__ == "__main__":
    main()
