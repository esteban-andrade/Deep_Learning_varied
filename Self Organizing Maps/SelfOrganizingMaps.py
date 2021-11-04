import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense


def importDataSet(dataset_path):

    dataset = pd.read_csv(dataset_path)
    # we choose all but the last one which is the class attribute
    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, -1].values  # we choose the latest attribute

    return X, Y, dataset
    """
     We are only taking these for the final classification but in the end we will only use X
    """


def FeatureScaling(data):
    X = data[0]
    y = data[1]
    sc = MinMaxScaler(feature_range=(0, 1))

    X = sc.fit_transform(X)

    # Training SOM
    """
    x,y = Dimentions of SOM (arbitrary dimension but shouldnt be small)

    input length = number of features in X
    sigma = radius of neightbours
    The bigger the learning rate the faster it will convergue and the smaller the more time it will take to finish learning
    decay function can be used to improve convergence
    """

    som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
    som.random_weights_init(X)
    som.train_random(data=X, num_iteration=100)

    """VISUALIZE"""

    bone()

    # distances are added to the map as the color mean distances
    pcolor(som.distance_map().T)
    colorbar()
    markers = ['o', 's']  # circle and square
    colors = ['r', 'g']
    for i, x in enumerate(X):
        w = som.winner(x)
        plot(w[0] + 0.5,  # the plus 0.5 slais to put in on the center of the square
             w[1] + 0.5,
             markers[y[i]],
             markeredgecolor=colors[y[i]],
             markerfacecolor='None',
             markersize=10,
             markeredgewidth=2)
    show()

    return som


def Prediction(model, data):
    X = data[0]
    som = model
    sc = MinMaxScaler(feature_range=(0, 1))

    X = sc.fit_transform(X)
    mappings = som.win_map(X)
    frauds = np.concatenate((mappings[(4, 3)], mappings[(8, 4)]), axis=0)
    frauds = sc.inverse_transform(frauds)

    print('Fraud Customer IDs')

    for i in frauds[:, 0]:
        print(int(i))

    return frauds


def CreateAndTrainNeuralNetwork(data):
    ann = tf.keras.models.Sequential()

    # Adlayers
    ann.add(tf.keras.layers.Dense(units=2, activation="relu"))
    # add output layer
    ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    """TRAIN NN"""
    ann.compile(optimizer="adam", loss="binary_crossentropy",
                metrics=["accuracy"])

    ann.fit(data[0], data[1], batch_size=1, epochs=15)
    return ann


def AdjustDataNN(data, SOM_Data):
    # matrix of features
    customers = data[2].iloc[:, 1:].values

    # we need a dependable variable for supervised learning
    is_fraud = np.zeros(len(data[2]))  # initialise vector
    for i in range(len(data[2])):
        if data[2].iloc[i, 0] in SOM_Data:  # check if customer ID has been in fraud
            is_fraud[i] = 1  # Assign to 1 if there is a match

    # FEATURE SCALING
    sc = StandardScaler()
    customers = sc.fit_transform(customers)

    return customers, is_fraud


def ANNPRedict(model, X_variable, data):
    y_prediction = model.predict(X_variable)
    # we concatonatre the customer ID and the predictions
    y_prediction = np.concatenate((data[2].iloc[:, 0:1].values,y_prediction), axis=1)

    # sort customer by provabability.
    # we sort first column which is index 1
    y_prediction = y_prediction[y_prediction[:, 1].argsort()]

    return y_prediction


def main():
    data_set_path = "Credit_Card_Applications.csv"

    data = importDataSet(data_set_path)

    som = FeatureScaling(data)
    frauds = Prediction(som, data)

    """" DEEP Learning SUpervised"""
    customers_data = AdjustDataNN(data, frauds)

    ANN = CreateAndTrainNeuralNetwork(customers_data)
    np.set_printoptions(suppress=True)
    print(ANNPRedict(ANN,customers_data[0],data))


if __name__ == "__main__":
    main()
