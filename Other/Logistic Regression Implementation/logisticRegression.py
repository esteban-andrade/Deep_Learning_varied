import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap


def getDatasetValues(dataset):
    X = dataset.iloc[:, :-1].values  # get all values but the last one
    Y = dataset.iloc[:, -1].values  # get last value as dependable value

    return X, Y


def divideDateset(var1, var2):
    # TEST SIZE = PERCENTAGE OOF OBSERVATION THAT WILL GO TO TESTS VARIABLES
    X_train, X_test, y_train, y_test = train_test_split(
        var1, var2, test_size=0.25, random_state=0)
    return X_train, X_test, y_train, y_test


def featureScaling(train_var, test_var):
    scaler = StandardScaler()

    train_var = scaler.fit_transform(train_var)
    test_var = scaler.transform(test_var)
    return train_var, test_var, scaler


# Training the Logistic Regression model on the Training set
def trainingModel(x, y):
    model = LogisticRegression(random_state=0)
    model.fit(x, y)
    return model


def confusionMatrix(test, pred):
    matrix = confusion_matrix(test, pred)
    accuracy = accuracy_score(test, pred)

    return matrix, accuracy


def visualiser(sc, X_train, y_train, classifier):
    X_set, y_set = sc.inverse_transform(X_train), y_train
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25),
                         np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25))
    plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c=ListedColormap(('red', 'green'))(i), label=j)
    plt.title('Logistic Regression (Training set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()


def main():
    dataset = pd.read_csv('Social_Network_Ads.csv')
    X, Y = getDatasetValues(dataset=dataset)
    X_train, X_test, y_train, y_test = divideDateset(X, Y)
    X_train, X_test, sc = featureScaling(X_train, X_test)
    classifier = trainingModel(X_train, y_train)

    # predict new result

    # input mmust be in [[]]
    print(classifier.predict(sc.transform([[30, 87000]])))

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    print(np.concatenate((y_pred.reshape(len(y_pred), 1),
          y_test.reshape(len(y_test), 1)), 1))

    confusion_matrix, accuracy = confusionMatrix(y_test, y_pred)
    print(confusion_matrix)
    print(str(accuracy*100) + "%")

    # Visualising the Training set results
    visualiser(sc,X_train,y_train,classifier)

    # Visualising the Test set results
    visualiser(sc,X_test,y_test,classifier)

if __name__ == "__main__":
    main()
