import numpy as np
import matplotlib as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def getDatasetValues(dataset):
    X = dataset.iloc[:, :-1].values  # get all values but the last one
    Y = dataset.iloc[:, -1].values  # get last value as dependable value

    return X, Y


def adjustMissingData(parameter):
    # REPLACE MISSING DATA WITH THE AVERAGE OF ALL
    input = SimpleImputer(missing_values=np.nan, strategy="mean")
    # will look at missing values and compute mean
    input.fit(parameter[:, 1:3])
    # will replace missing values with the meam
    # ensure we dont use string columns
    parameter[:, 1:3] = input.transform(parameter[:, 1:3])

    return parameter


def encodeIndependantVariable(variable):
    encoder = ColumnTransformer(
        # Transformers.
        # 1. type of Transformation --> ENcoding
        # 2. Type of encoding -> OneHotEncoder
        # 3. Index of columns

        # passthrough means we willnot encode the other columns and we will leave them
        transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough")

    variable = np.array(encoder.fit_transform(variable))
    return variable


def encodeDependableVariable(variable):

    encoder = LabelEncoder()
    variable = encoder.fit_transform(variable)
    return variable


def divideDateset(var1, var2):
    # TEST SIZE = PERCENTAGE OOF OBSERVATION THAT WILL GO TO TESTS VARIABLES
    X_train, X_test, y_train, y_test = train_test_split(
        var1, var2, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test


def featureScaling(train_var, test_var):
    scaler = StandardScaler()
    # we dont apply features on the dummy encoded string values
    train_var[:, 3:] = scaler.fit_transform(train_var[:, 3:])
    test_var[:, 3:] = scaler.transform(test_var[:, 3:])
    return train_var, test_var


def main():
    dataset = pd.read_csv("Data.csv")
    x, y = getDatasetValues(dataset)
    x = adjustMissingData(x)
    x = encodeIndependantVariable(x)
    print(x)
    y = encodeDependableVariable(y)
    print(y)
    X_train, X_test, y_train, y_test = divideDateset(x, y)
    X_train, X_test = featureScaling(X_train, X_test)
    print(X_train)
    print(X_test)


if __name__ == "__main__":
    main()
