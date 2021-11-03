import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show


def importDataSet(dataset_path):

    dataset = pd.read_csv(dataset_path)
    # we choose all but the last one which is the class attribute
    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, -1].values  # we choose the latest attribute

    return X, Y
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
        plot(w[0] + 0.5, # the plus 0.5 is to put in on the center of the square
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
    frauds = np.concatenate((mappings[(1, 1)], mappings[(4, 1)]), axis=0)
    frauds = sc.inverse_transform(frauds)

    print('Fraud Customer IDs')

    for i in frauds[:, 0]:
        print(int(i))


def main():
    data_set_path = "Credit_Card_Applications.csv"

    data = importDataSet(data_set_path)

    som = FeatureScaling(data)
    Prediction(som, data)


if __name__ == "__main__":
    main()
