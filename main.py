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
        teachingInput = teachingInput.reshape((teachingInput.shape[0], 1))
        self.num_of_outputAttr = teachingInput.shape[1]

        self.dataset = np.concatenate((inputs, teachingInput), axis=1)


    def checkSettings(self):
        print "the setting of NN is:"
        print "the inputNeurons = %d" % self.inputNeurons.shape[0]
        print "the hiddenNeurons = %d" % self.hiddenNeurons.shape[0]
        print "the outputNeurons = %d" % self.outputNeurons.shape[0]

        print "learning rate is: %f" % self.r
        print "momentum is %f" % self.m
        print "learning_criterion is %f" % self.learning_criterion

        print "num of pattern is: %d" % self.dataset.shape[0]
        print "num_of_inputAttr is %d" % self.num_of_inputAttr
        print "num_of_outputAttr is %d" % self.num_of_outputAttr
        
        print "the dataset is: "
        print self.dataset

    def checkInitializedWeightsAndBiases(self):
        print "W_1 : "
        print nn.W_1

        print "W_2 : "
        print nn.W_2

        print "B_h: "
        print nn.B_h

        print "B_o:"
        print nn.B_o


    # initialize weights between layers, including biases
    def initializeWeights(self):
        # W_1 is the weight between input layter and hidden layter, j row for j hidden neurons
        # i column of i input neurons
        self.W_1 = np.zeros(shape=(self.hiddenNeurons.shape[0], self.inputNeurons.shape[0]))

        # W_2 is k row for output neurons, j column for j hidden neurons
        self.W_2 = np.zeros(shape=(self.outputNeurons.shape[0], self.hiddenNeurons.shape[0]))
        
        for j in xrange(0, self.W_1.shape[0]):
            for i in xrange(0, self.W_1.shape[1]):
                self.W_1[j][i] = random.uniform(0.1, 0.7)

        for k in xrange(0, self.W_2.shape[0]):
            for j in xrange(0, self.W_2.shape[1]):
                self.W_2[k][j] = random.uniform(0.1, 07)

        
        # B_h is biases for hidden neurons
        self.B_h = np.zeros(shape=(self.hiddenNeurons.shape[0], 1))
        for j in xrange(0, self.B_h.shape[0]):
            self.B_h[j] = random.uniform(0.1, 0.7)

        # B_o is baises for output neurons
        self.B_o = np.zeros(shape=(self.outputNeurons.shape[0], 1))
        for k in xrange(0, self.B_o.shape[0]):
            self.B_o[k] = random.uniform(0.1, 0.7)

        # Need to initialize the previous changes of weights
        self.delta_W_1 = np.zeros(shape=(self.hiddenNeurons.shape[0], self.inputNeurons.shape[0]))
        self.delta_W_2 = np.zeros(shape=(self.outputNeurons.shape[0], self.hiddenNeurons.shape[0]))
        self.delta_B_h = np.zeros(shape=(self.hiddenNeurons.shape[0], 1))
        self.delta_B_o = np.zeros(shape=(self.outputNeurons.shape[0], 1))

    # this function compute the change of states for a given input pattern
    def compute_forward(self, input):
        # the first layer's state will be setted directly by input
        for i in xrange(0, self.inputNeurons.shape[0]):
            self.inputNeurons[i] = input[i]

        # the second / third layer's states = sum ( w * (output of previous layer + bias = 1)) go through
        # activation function

        # for second layer, the hidden neurons
        for j in xrange(0, self.hiddenNeurons.shape[0]):
            sum = 0.0
            for i  in xrange(0, self.inputNeurons.shape[0]):
                sum += (self.inputNeurons[i] * self.W_1[j][i])
                
            sum += self.B_h[j]
            self.hiddenNeurons[j] = (1.0 / (1.0 + math.exp(-sum)))

        # simlar for third layer, the output neurons
        for k in xrange(0, self.outputNeurons.shape[0]):
            sum = 0.0
            for j in xrange(0, self.hiddenNeurons.shape[0]):
                sum += (self.hiddenNeurons[j] * self.W_2[k][j])
                
            sum += self.B_o[k]
            self.outputNeurons[k] = (1.0 / (1.0 + math.exp(-sum)))

    # function compute the back-propagation based on errors
    def computeBackpropagation(self, backErrors):
        # save the orignal weights, since later they will be changed
        W_2_save = np.copy(self.W_2)

        # for second layer weights
        deltaPKs = np.zeros(shape=(self.outputNeurons.shape[0], 1))
        for k in xrange(self.outputNeurons.shape[0]):
            deltaPKs[k] = backErrors[k] * self.outputNeurons[k] * (1.0 - self.outputNeurons[k])

            for j in xrange(self.hiddenNeurons.shape[0]):
                self.W_2[k][j] = W_2_save[k][j] + (self.r * deltaPKs[k] * self.hiddenNeurons[j]) + self.m * self.delta_W_2[k][j]
                self.B_o[k] += ((self.r * deltaPKs[k] * 1.0) + self.m * self.delta_B_o[k])
                # record down the changes of weights and bias for Momentum
                self.delta_W_2[k][j] = self.r * deltaPKs[k] * self.hiddenNeurons[j]
                self.delta_B_o[k] = self.r * deltaPKs[k] * 1.0

        # for first layer weights
        for j in xrange(self.hiddenNeurons.shape[0]):
            sumFromOutputLayer = 0.0
            for i in xrange(self.inputNeurons.shape[0]):
                sumFromOutputLayer += (deltaPKs[k] * W_2_save[k][j])

            deltaPJ = self.hiddenNeurons[j] * (1 - self.hiddenNeurons[j]) * sumFromOutputLayer
            for i in xrange(self.inputNeurons.shape[0]):
                self.W_1[j][i] += (self.r * deltaPJ * self.inputNeurons[i] + self.m * self.delta_W_1[j][i])
                self.B_h[j] += (self.r * deltaPJ * 1.0 + self.m * self.delta_B_h[j])
                self.delta_W_1[j][i] = self.r * deltaPJ * self.inputNeurons[i]
                self.delta_B_h[j] = self.r * deltaPJ * 1.0

    # functions do the trainning on dataset
    def train(self, training_set, epoches):
        epoch = 0

        while(epoches > 0):
            # for each epoch
            backErrors = np.zeros(shape=(self.outputNeurons.shape[0],1))
            
            # shuffle the dataset for each epoch training
            np.random.shuffle(training_set)
            sum = 0.0

            for i in xrange(0, training_set.shape[0]):
                # for each row, slice off the corresponding input part
                self.compute_forward(training_set[i][:self.num_of_inputAttr])

                for k in xrange(0, self.num_of_outputAttr):
                    # self.dataset[i][k + self.num_of_inputAttr] is the corresponding output part
                    err = training_set[i][k + self.num_of_inputAttr] - self.outputNeurons[k]
                    sum += (err * err)
                    backErrors[k] = err
                    
                self.computeBackpropagation(backErrors)

            popErr = sum / (self.num_of_outputAttr * training_set.shape[0])
            epoch += 1

            if epoch % 100 == 0:
                print "epoch = %d, popErr = %f" % (epoch, popErr)
                
            epoches -= 1
            if popErr < self.learning_criterion:
                break


# a function used for prompting to guid use to test training result
def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        print "\n==="
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


    
if __name__ == '__main__':

    # load params and data from .txt file in the current directory
    params = np.loadtxt('param.txt')
    inputs = np.loadtxt('input.txt')
    teachingInput = np.loadtxt('teaching_input.txt')

    nn = NN(params, inputs, teachingInput)

    nn.initializeWeights()
    nn.checkInitializedWeightsAndBiases()
    nn.checkSettings()

    # === training phrase ===
    # divide data set into training_set and testing_set
    # only dived data when it is on Iris dataset
    if nn.dataset.shape[0] == 150:
        np.random.shuffle(nn.dataset)
        # make the first 0~99 patters as testing pattern
        training_set = nn.dataset[:100]
        # make the 100~149 patterns as testing pattern
        testing_set = nn.dataset[-50:150]
    else:
        training_set = nn.dataset
        testing_set = nn.dataset

    nn.train(training_set, 2000000)

    # === testing phrase ===
    while True:
        if query_yes_no("Pick a random patter for testing?"):
            # randomly pick a pattern from testing_set
            np.random.shuffle(testing_set)
            pattern = testing_set[0]
            input = pattern[:nn.num_of_inputAttr]
            teaching_input = pattern[-1*nn.num_of_outputAttr:nn.dataset.shape[0]]

            print "the input is: "
            print input
            print "the teaching input is: "
            print teaching_input

            print "the output from NN is"
            nn.compute_forward(input)
            print nn.outputNeurons
        else:
            sys.exit(0)


