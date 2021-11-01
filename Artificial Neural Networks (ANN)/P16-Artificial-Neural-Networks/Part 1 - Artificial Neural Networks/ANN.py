'''
Steps for creating ANN
1. Data Preparation
2. Building ANN
3. Training ANN
4. Making prediction and Evaluate Model

'''
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder  # for label enconding
# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split  # Used to get test Models
from sklearn.preprocessing import StandardScaler  # Scale
from sklearn.metrics import confusion_matrix, accuracy_score


def DataPreparation():

    dataset = pd.read_csv("Churn_Modelling.csv")
    # all the features from the 4th column to the end
    X = dataset.iloc[:, 3:-1].values
    Y = dataset.iloc[:, -1].values
    # print(X)

    # Enconding Categories and Data
    le = LabelEncoder()
    X[:, 2] = le.fit_transform(X[:, 2])   # label enconde Gender
    # print(X)

    # Encode Geography column
    ct = ColumnTransformer(
        transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
    X = np.array(ct.fit_transform(X))
    """The countru names were encoded to number sequence such as 1.0 0.0 0.0"""
    # print(X)

    # Split Data into Training Set and TestSet
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=1)

    # print(X_train)
    # print(X_test)
    # print(y_train)
    # print(y_test)

    ## FEATURE SCALING##
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    # print(X_train)
    # print(X_test)
    return X_train, X_test, y_train, y_test


def Building_ANN():

    ann = tf.keras.models.Sequential()  # add sequential Model

    # add Innput and Hidden laters
    # units = number of input units
    ann.add(tf.keras.layers.Dense(units=6, activation="relu"))

    # add second layer
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

    # add output layer
    ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    return ann


def Train_ANN(data, model):
    # pass parameters
    ann = model
    X_train, X_test, y_train, y_test = data

    # compile ANN
    ann.compile(optimizer="adam", loss="binary_crossentropy",
                metrics=["accuracy"])
    """
    * adam optimizer performs stochastic gradient Decent
    * loss : for binary outcome it must be binary_crossentroply
     for non binary iy must be category_crossentropy and the activation functrion should be softmax
    * we can choose different metrics
    """
    # Training ANN
    ann.fit(X_train, y_train, batch_size=32, epochs=100)
    return ann


def predict_single_entry(model, fitting):
    ann = model
    sc = StandardScaler()
    # always 2D array
    sc.fit(fitting)
    value = sc.transform(
        [[1.0, 0.0, 0.0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])
    print(ann.predict(value) > 0.5)


def predict_tests(model, X_test, y_test):
    ann = model
    y_pred = ann.predict(X_test)
    y_pred = (y_pred > 0.5)
    print(np.concatenate((y_pred.reshape(len(y_pred), 1),
          y_test.reshape(len(y_test), 1)), 1))
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(accuracy_score(y_test, y_pred))


def main():
    print(tf.__version__)
    data = DataPreparation()
    ann = Building_ANN()

    Train_ANN(data, ann)

    print("\n---------PREDICTION-----------\n")
    predict_single_entry(ann, data[0])

    print("\n---------PREDICTION TEST-----------\n")
    predict_tests(ann, data[1], data[3])


if __name__ == "__main__":
    main()
