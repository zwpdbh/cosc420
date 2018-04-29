import numpy as np
import os
import random
import math
import sys


class NN:
    """Hold the input and teaching input"""

    def __init__(self, params, inputs, teachingInput):
        self.inputNeurons = np.zeros(shape=(int(params[0]), 1))
        self.hiddenNeurons = np.zeros(shape=(int(params[1]),1))
        self.outputNeurons = np.zeros(shape=(int(params[2]), 1))

        self.r = params[3]
        self.m = params[4]
        self.learning_criterion = params[5]
        
        # convert data into array shape
        inputs = np.asarray(inputs)
        self.num_of_inputAttr = inputs.shape[1]
        
        teachingInput = np.asarray(teachingInput)
        teachingInput = teachingInput.reshape((teachingInput.shape[0], teachingInput.shape[1]))
        self.num_of_outputAttr = teachingInput.shape[1]

        self.dataset = np.concatenate((inputs, teachingInput), axis=1)

    # initialize weights between layers, including biases
    def initializeWeights(self):
        max_random_w = 0.95
        # W_1 is the weight between input layter and hidden layter, j row for j hidden neurons
        # i column of i input neurons
        self.W_1 = np.zeros(shape=(self.hiddenNeurons.shape[0], self.inputNeurons.shape[0]))

        # W_2 is k row for output neurons, j column for j hidden neurons
        self.W_2 = np.zeros(shape=(self.outputNeurons.shape[0], self.hiddenNeurons.shape[0]))
        
        for j in range(0, self.W_1.shape[0]):
            for i in range(0, self.W_1.shape[1]):
                self.W_1[j][i] = random.uniform(0.1, max_random_w)

        for k in range(0, self.W_2.shape[0]):
            for j in range(0, self.W_2.shape[1]):
                self.W_2[k][j] = random.uniform(0.1, max_random_w)

        
        # B_h is biases for hidden neurons
        self.B_h = np.zeros(shape=(self.hiddenNeurons.shape[0], 1))
        for j in range(0, self.B_h.shape[0]):
            self.B_h[j] = random.uniform(0.1, max_random_w)

        # B_o is baises for output neurons
        self.B_o = np.zeros(shape=(self.outputNeurons.shape[0], 1))
        for k in range(0, self.B_o.shape[0]):
            self.B_o[k] = random.uniform(0.1, max_random_w)

        # Need to initialize the previous changes of weights
        self.delta_W_1 = np.zeros(shape=(self.hiddenNeurons.shape[0], self.inputNeurons.shape[0]))
        self.delta_W_2 = np.zeros(shape=(self.outputNeurons.shape[0], self.hiddenNeurons.shape[0]))
        self.delta_B_h = np.zeros(shape=(self.hiddenNeurons.shape[0], 1))
        self.delta_B_o = np.zeros(shape=(self.outputNeurons.shape[0], 1))



    # this function compute the change of states for a given input pattern
    def compute_forward(self, input):
        # the first layer's state will be setted directly by input
        for i in range(0, self.inputNeurons.shape[0]):
            self.inputNeurons[i] = input[i]

        # the second / third layer's states = sum ( w * (output of previous layer + bias = 1)) go through
        # activation function

        # for second layer, the hidden neurons
        for j in range(0, self.hiddenNeurons.shape[0]):
            sum = 0.0
            for i  in range(0, self.inputNeurons.shape[0]):
                sum += (self.inputNeurons[i] * self.W_1[j][i])
                
            sum += self.B_h[j]
            self.hiddenNeurons[j] = (1.0 / (1.0 + math.exp(-sum)))

        # simlar for third layer, the output neurons
        for k in range(0, self.outputNeurons.shape[0]):
            sum = 0.0
            for j in range(0, self.hiddenNeurons.shape[0]):
                sum += (self.hiddenNeurons[j] * self.W_2[k][j])
                
            sum += self.B_o[k]
            self.outputNeurons[k] = (1.0 / (1.0 + math.exp(-sum)))

    # function compute the back-propagation based on errors
    def computeBackpropagation(self, backErrors):
        # save the orignal weights, since later they will be changed
        W_2_save = np.copy(self.W_2)

        # for second layer weights
        deltaPKs = np.zeros(shape=(self.outputNeurons.shape[0], 1))
        for k in range(self.outputNeurons.shape[0]):
            deltaPKs[k] = backErrors[k] * self.outputNeurons[k] * (1.0 - self.outputNeurons[k])

            for j in range(self.hiddenNeurons.shape[0]):
                self.W_2[k][j] = W_2_save[k][j] + (self.r * deltaPKs[k] * self.hiddenNeurons[j]) + self.m * self.delta_W_2[k][j]
                self.B_o[k] += ((self.r * deltaPKs[k] * 1.0) + self.m * self.delta_B_o[k])
                # record down the changes of weights and bias for Momentum usage: learning_rate * delta_wji (n - 1)
                self.delta_W_2[k][j] = self.r * deltaPKs[k] * self.hiddenNeurons[j]
                self.delta_B_o[k] = self.r * deltaPKs[k] * 1.0

        # for first layer weights
        for j in range(self.hiddenNeurons.shape[0]):
            sumFromOutputLayer = 0.0
            for i in range(self.inputNeurons.shape[0]):
                sumFromOutputLayer += (deltaPKs[k] * W_2_save[k][j])

            deltaPJ = self.hiddenNeurons[j] * (1 - self.hiddenNeurons[j]) * sumFromOutputLayer
            for i in range(self.inputNeurons.shape[0]):
                self.W_1[j][i] += (self.r * deltaPJ * self.inputNeurons[i] + self.m * self.delta_W_1[j][i])
                self.B_h[j] += (self.r * deltaPJ * 1.0 + self.m * self.delta_B_h[j])
                # record down the changes of weights and bias for Momentum usage: learning_rate * delta_wji (n - 1)
                self.delta_W_1[j][i] = self.r * deltaPJ * self.inputNeurons[i]
                self.delta_B_h[j] = self.r * deltaPJ * 1.0

    # train one epoch for training_set
    def train(self, training_set):
        backErrors = np.zeros(shape=(self.outputNeurons.shape[0], 1))

        # shuffle the dataset for each epoch training
        np.random.shuffle(training_set)
        sum = 0.0

        for i in range(0, training_set.shape[0]):
            # for each row, slice off the corresponding input part
            self.compute_forward(training_set[i][:self.num_of_inputAttr])
            for k in range(0, self.num_of_outputAttr):
                # self.dataset[i][k + self.num_of_inputAttr] is the corresponding output part
                err = training_set[i][k + self.num_of_inputAttr] - self.outputNeurons[k]
                sum += (err * err)
                backErrors[k] = err
            self.computeBackpropagation(backErrors)

        popErr = sum / (self.num_of_outputAttr * training_set.shape[0])
        return popErr



