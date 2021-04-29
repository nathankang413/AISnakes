import math
import numpy as np
from random import random
import matplotlib.pyplot as plt
import time

"""
Names / definitions:
- Weight = a single weight in a layer
- Layer = one layer in a dense NN
- Theta = a list of all the layers in a dense NN (just the numbers, no functionality)
- Brain = a dense NN with 18 inputs and 3 outputs (adds functionality to theta)
"""


def sigmoid(z):
    return 1 / (1 + math.exp(z))


sigmoid_vect = np.vectorize(sigmoid)


def rand_init_layer(l_in, l_out, epsilon):
    return np.random.rand(l_out, 1 + l_in) * (2 * epsilon) - epsilon


class SnakeBrain:

    """
    A dense NN configured to play snake
    Generally should have 18 inputs and 3 outputs to be compatible with the game
    """

    def __init__(self, theta=None, sizes=(), epsilon=1, file_name=''):

        """
        Creates a new SnakeBrain.
        Brain can be initialized with set layers/weights, set layer sizes, or randomly with an epsilon value

        :param theta: when given, brain uses these layers/weights
        :param sizes: if theta is not provided, sizes are used to randomly initialize weights, default (18, 10, 10, 3)
        :param epsilon: controls the randomness for random initialization
        :param file_name: when given, the file containing the layers/weights
        """

        self.epsilon = epsilon
        if file_name != '':
            self.read_brain(file_name)
        elif theta is not None:
            self.theta = theta
            self.sizes = []
            for layer in self.theta:
                self.sizes.append(layer.shape[1]-1)
            self.sizes.append(self.theta[-1].shape[0])
            self.sizes = tuple(self.sizes)
        else:
            if sizes == ():
                self.sizes = (18, 10, 10, 3)
            else:
                self.sizes = sizes

            self.rand_init_theta()

    def rand_init_theta(self):
        """
        Randomly initializes theta using sizes and epsilon
        """

        self.theta = []
        for i in range(len(self.sizes) - 1):
            self.theta.append(rand_init_layer(self.sizes[i], self.sizes[i + 1], self.epsilon))

    def predict(self, data):
        """
        Generates a prediction vector with size (3, ) of probabilities for each move

        :param data: a vector of data with size (18, )
        :return: a prediction vector of with (3, )
        """
        if len(data) != self.sizes[0]:
            print(f'Error: Data is of length {len(data)}, but brain takes {self.sizes[0]} inputs.')
            return None

        a = np.concatenate((np.array([1]), np.array(data)))
        for i in range(len(self.theta)):
            z = np.dot(self.theta[i], a)
            a = np.concatenate((np.array([1]), sigmoid_vect(z)))
        h = a[1:]
        return h

    def save_brain(self, file_name):
        """
        Save the brain to the given file in hexidecimal format

        :param file_name: file_name to store theta
        """

        layer_bytes = [arr.tobytes() for arr in self.theta]
        layer_strings = [l_bytes.hex() for l_bytes in layer_bytes]
        brain_string = '\n'.join(layer_strings)

        size_str = ','.join([str(size) for size in self.sizes])
        save_str = size_str + '\n' + brain_string

        with open(file_name, 'w') as file:
            file.write(save_str)

    def read_brain(self, file_name):
        """
        reads theta from the given file and initializes brain

        :param file_name: file from which to read theta
        """

        with open(file_name, 'r') as file:
            brain_string = file.read()
        line_strings = brain_string.split('\n')
        size_str = line_strings[0]
        sizes_list = [int(size) for size in size_str.split(',')]
        self.sizes = tuple(sizes_list)

        layer_strings = line_strings[1:]
        layer_bytes = [bytes.fromhex(l_string) for l_string in layer_strings]

        unshaped_theta = [np.frombuffer(l_bytes, dtype=float) for l_bytes in layer_bytes]
        self.theta = []
        for i in range(len(unshaped_theta)):
            layer = np.reshape(unshaped_theta[i], (self.sizes[i+1], self.sizes[i]+1))
            self.theta.append(layer)

    def copy_and_mutate(self, mut_chance, mut_size):
        """
        Generates a slightly mutated clone without changing the original

        :param mut_chance: chance of mutation for each weight
        :param mut_size: maximum size of mutation for each weight
        :return: a SnakeBrain slightly different from the original
        """

        new_theta = []
        for layer in self.theta:
            old_shape = layer.shape
            theta_list = layer.flatten().tolist()
            for i in range(len(theta_list)):
                if random() > mut_chance:
                    theta_list[i] += random() * 2 * mut_size - mut_size
            new_layer = np.array(theta_list).reshape(old_shape)
            new_theta.append(new_layer)
        return SnakeBrain(theta=new_theta)


def rand_gen_brains(num_brains=10, sizes=(), epsilon=1):

    """
    Randomly generates a list of SnakeBrain's with the given sizes and epsilon

    :param num_brains: the number of SnakeBrain's to generate
    :param sizes: the sizes of the layers in each brain
    :param epsilon: controls the randomness of each brain
    :return: a list of SnakeBrain's
    """

    brains = []
    for _ in range(num_brains):
        brains.append(SnakeBrain(sizes=sizes, epsilon=epsilon))
    return brains


def get_top_brains(brains, scores, frac_keep):
    """
    Given a list of brains and their scores, returns the best fraction of brains

    :param brains: the list of brains to evaluate
    :param scores: the scores of the brains
    :param frac_keep: the fraction of brains to keep between 0 and 1
    :return: a list of brains of size frac_keep * list(brains)
    """

    num_keep = int(frac_keep * len(scores))
    sorted_scores = sorted(scores)
    threshold = sorted_scores[-num_keep]
    scores_dict = {v: k for v, k in enumerate(scores)}

    top_brains = []
    for key in scores_dict:
        if scores_dict[key] >= threshold:
            top_brains.append(brains[key])
    return top_brains





