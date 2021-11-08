
"""
k step contrastive divergence algorithm
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


def DataProcessing():
    # movies = pd.read_csv("ml-1m/movies.dat", sep="::",  # engine is to ensure works with python and encoding is special charatecters that utf8 cant handle
    #                      header=None, engine="python", encoding="latin-1")

    # users = pd.read_csv("ml-1m/users.dat", sep="::",
    #                     header=None, engine="python", encoding="latin-1")

    # ratings = pd.read_csv("ml-1m/ratings.dat", sep="::",
    #                       header=None, engine="python", encoding="latin-1")

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
            id_movies = data[:, 1][data[:, 0] == id_users]
            id_ratings = data[:, 2][data[:, 0] == id_users]
            ratings = np.zeros(nb_movies)
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

    """
    CONVERT Ratings into Binary Rating
    These will be the input of te Boltzman Machines
    """
# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
    # movies that are not rated to customers. Not a rating giving
    training_set[training_set == 0] = -1
    training_set[training_set == 1] = 0
    training_set[training_set == 2] = 0
    training_set[training_set >= 3] = 1
    test_set[test_set == 0] = -1
    test_set[test_set == 1] = 0
    test_set[test_set == 2] = 0
    test_set[test_set >= 3] = 1

    return training_set, test_set


class RBM():  # nv Number of Visible Nodes. nh number of hidden nodes
    def __init__(self, nv, nh) -> None:
        self.W = torch.randn(nh, nv)  # weights
        # initialise the Bias
        self.a = torch.randn(1, nh)  # bias for hideen nodes given visible
        self.b = torch.randn(1, nv)  # bias for visible nodes

        """FUNCTION TO SAMPLE  HIDEN NODDES given visible nodes with Gibbs Sampling"""

    def sample_h(self, x):  # x is number of visible neurons

        # probability of H given v
        w_times_x = torch.mm(x, self.W.t())
        # activation function
        activation = w_times_x+self.a.expand_as(w_times_x)
        p_h_given_v = torch.sigmoid(activation)
        samples_hidden_neurons = torch.bernoulli(p_h_given_v)

        return p_h_given_v, samples_hidden_neurons

    def sample_v(self, y):
        w_times_y = torch.mm(y, self.W)
        # activation function
        activation = w_times_y+self.b.expand_as(w_times_y)
        p_v_given_h = torch.sigmoid(activation)
        samples_visible_neurons = torch.bernoulli(p_v_given_h)

        return p_v_given_h, samples_visible_neurons

    # function with contrastive divergence
    # v0 input vector
    # vk visible nodes obtained after k samplings
    # ph0 = vector of probability that first iteration node is equal to one
    # phk = vector of probs given  after k sampling giving vbisibles nodes vk
    def train(self, v0, vk, ph0, phk):
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0-vk), 0)
        self.a += torch.sum((ph0-phk), 0)


def main():
    training_set, test_set, nb_users, nb_movies = DataProcessing()
    training_set_converted, test_set_converted = DataConversion(
        training_set, test_set)

    nv = len(training_set_converted[0])
    nh = 100
    batch_size = 100

    rbm = RBM(nv, nh)

    """TRAINNG RBM"""
    number_epochs = 10
    for epoch in range(1, number_epochs+1):
        train_loss = 0
        counter = 0.0

        for id_user in range(0, nb_users-batch_size, batch_size):
            # vk Input vector
            vk = training_set_converted[id_user:id_user+batch_size]
            # v0 initial values
            v0 = training_set_converted[id_user:id_user+batch_size]
            # initial probabilities
            # the _ is used to retrieve only the first element
            ph0, _ = rbm.sample_h(v0)
            for i in range(10):
                _, hk = rbm.sample_h(vk)
                _, vk = rbm.sample_v(hk)
                # we dont learn where there is no rating
                vk[v0 < 0] = v0[v0 < 0]  # freeze cell that use -1

            phk, _ = rbm.sample_h(vk)  # applied to the last sample

            rbm.train(v0, vk, ph0, phk)
            train_loss += torch.mean(torch.abs(v0[v0 >= 0]-vk[v0 >= 0]))

            # RMSE here
            #train_loss += np.sqrt(torch.mean((v0[v0 >= 0] - vk[v0 >= 0])**2))
            # Average Distance
            # train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))  # Average Distance here
            counter += 1

        print('epoch: '+str(epoch)+' loss: '+str(train_loss/counter))

    print("\nTESTING\n")
    """"TESTING"""
    test_loss = 0
    counter = 0.0

    for id_user in range(nb_users):
        # input to activate the hideen neurons
        v = training_set_converted[id_user:id_user+1]
        vt = test_set_converted[id_user:id_user+1]
        if len(vt[vt >= 0]) > 0:
            _, h = rbm.sample_h(v)
            _, v = rbm.sample_v(h)
            test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))
            # test_loss += np.sqrt(torch.mean((vt[vt>=0] - v[vt>=0])**2)) # RMSE here
            # Average Distance
            #  test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0])) # Average Distance here
            counter += 1

        print('test loss: '+str(test_loss/counter))


if __name__ == "__main__":
    main()
