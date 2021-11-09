import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# torch.cuda.is_available = lambda: False
torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def DataProcessing():
    training_set = pd.read_csv(
        "ml-100k/u1.base", delimiter="\t")  # delimeter same as sep
    # dtype specific the type of array
    training_set = np.array(training_set, dtype="int")
    test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t')
    test_set = np.array(test_set, dtype='int')

    """ get humber of users and Movies
    Conver these into matrixes
    the matrices will have same number of users and movies same size, each cell will have the rating by user
    """
    nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
    nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))

    """ Convert Data into array matrix"""
    def convert(data):
        new_data = []
        for id_users in range(1, nb_users+1):
            # condition to match the users
            # takes all the movies ID of the users
            id_movies = data[:, 1][data[:, 0] == id_users]
            id_ratings = data[:, 2][data[:, 0] == id_users]
            ratings = np.zeros(nb_movies)
            # replace zeros with real ratings
            ratings[id_movies-1] = id_ratings
            new_data.append(list(ratings))

        return new_data

    training_set = convert(training_set)
    test_set = convert(test_set)
    return training_set, test_set, nb_users, nb_movies


def DataConversion(training_set, test_set):
    """ CONVERT INTO TORCH TENSOR"""
    training_set = torch.FloatTensor(training_set)
    test_set = torch.FloatTensor(test_set)
    training_set = training_set.cuda()
    test_set = test_set.cuda()
    return training_set, test_set


class StackAutoEncoder(nn.Module):
    def __init__(self, number_data) -> None:
        # super is used to inherit method from parent class
        super(StackAutoEncoder, self).__init__()
        # number of layers and featrures
        # number of feature, number of nodes
        self.connection_1 = nn.Linear(number_data, 20)
        self.connection_2 = nn.Linear(20, 10)

        # start of deconding part
        self.connection_3 = nn.Linear(10, 20)
        self.connection_4 = nn.Linear(20, number_data)

        # activation
        self.activation = nn.Sigmoid()

    """ ENCONDING AND DECONDING"""

    def propagation(self, x):
        x = self.activation(self.connection_1(x))
        x = self.activation(self.connection_2(x))
        x = self.activation(self.connection_3(x))
        x = self.connection_4(x)  # decoding
        return x


def main():

    print(device)
    training_set, test_set, nb_users, nb_movies = DataProcessing()
    training_set_converted, test_set_converted = DataConversion(
        training_set, test_set)

    Encoder = StackAutoEncoder(nb_movies)
    Encoder.to(device)
    criterion = nn.MSELoss()  # criterion for loss cost function
    optimizer = optim.RMSprop(Encoder.parameters(), lr=0.01, weight_decay=0.5)

    """TRAINING MODEL """

    nb_epoch = 200
    for epoch in range(1, nb_epoch+1):
        loss_counter = 0
        counter = 0.0
        for user_id in range(nb_users):
            # we create new dimension to make it a vector
            input = Variable(training_set_converted[user_id]).unsqueeze(0)
            target = input.clone()
            if torch.sum(target.data > 0) > 0:
                output = Encoder.propagation(input)
                target.require_grad = False  # ensures to not compute gradient with respect to target
                # to save memeory as this will not be required
                output[target == 0] = 0
                # real values and predicted values
                loss = criterion(output, target)
                # average of movies that were rated
                mean_corrector = nb_movies / \
                    float(torch.sum(target.data > 0)+1e-10)

                loss.backward()  # direction to update weigh

                loss_counter += torch.sqrt(loss.data*mean_corrector)
                counter += 1.
                optimizer.step()

        print('epoch: '+str(epoch)+'\tloss: ' + str(loss_counter/counter))

    """TESTING MODEL"""

    print("\nTESTING\n")

    test_loss = 0
    counter = 0.0
    for user_id in range(nb_users):
        input = Variable(training_set_converted[user_id]).unsqueeze(0)
        target = Variable(test_set_converted[user_id]).unsqueeze(0)
        if torch.sum(target.data > 0) > 0:
            output = Encoder.propagation(input)
            target.require_grad = False  # ensures to not compute gradient with respect to target
            # to save memeory as this will not be required
            output[target == 0] = 0
            # real values and predicted values
            loss = criterion(output, target)
            # average of movies that were rated
            mean_corrector = nb_movies / \
                float(torch.sum(target.data > 0)+1e-10)
            test_loss += torch.sqrt(loss.data*mean_corrector)
            counter += 1.
    print('loss: ' + str(loss_counter/counter))


if __name__ == "__main__":
    main()
