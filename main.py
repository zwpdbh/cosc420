import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork import NN
from datetime import datetime

class Controller:
    def __init__(self):
        self.accuracy_criteria = 0.9
        pass

    def show_menu(self):
        print("\nPlease input 0 - 9 to select:")
        print("1 : initialize")
        print("2 : teach 100 epochs")
        print("3 : teach until accuracy >= %.2f during testing" % self.accuracy_criteria)
        print("4 : teach to criteria")
        print("5 : randomly select one patter to test")
        print("6 : show weights")
        print("7 : run 100 test and collect training result")
        print("8 : check hidden units")
        print("9 : check settings without re-initialize the net")
        print("0 : quit")

    def checkSetting(self):
        print("The current setting is:")
        print("the number of inputNeurons = %d" % self.nn.inputNeurons.shape[0])
        print("the number of hiddenNeurons = %d" % self.nn.hiddenNeurons.shape[0])
        print("the number of outputNeurons = %d" % self.nn.outputNeurons.shape[0])
        print("learning rate = %.3f" % self.nn.r)
        print("momentum = %.3f" % self.nn.m)
        print("learning criterion = %.3f" % self.nn.learning_criterion)
        print("number of patterns = %d" % self.nn.dataset.shape[0])

    # for a given testing patter, check the corresponding hidden units, may needed for encoder/decoder
    def checkHiddenUnits(self):
        np.random.shuffle(self.testing_set)
        pattern = self.testing_set[0]
        input = pattern[:self.nn.num_of_inputAttr]
        i = self.nn.num_of_outputAttr
        j = pattern.shape[0]
        teaching_input = pattern[-i:j]

        print("\nthe input is: " + str(input))
        print("the teaching input is: " + str(teaching_input))
        self.nn.compute_forward(input)
        # the transpose and index slice is used to make display format be nice
        print("the output is: " + str(self.nn.outputNeurons.T[0]))
        print("the hidden units are: " + str(self.nn.hiddenNeurons.T[0]))

    def initialize(self):
        params = np.loadtxt('param.txt')
        inputs = np.loadtxt('input.txt')
        teachingInput = np.loadtxt('teaching_input.txt')

        self.test_T = 0.0
        self.test_F = 0.0
        self.accuracy = 0.0
        self.fit_criteria = 0.40
        self.nn = NN(params, inputs, teachingInput)
        self.nn.initializeWeights()
        self.popErr_record = []
        self.accuracy_record = []
        self.epoch_record = []

        print("Initialization complete, current settings are:")
        print("the number of inputNeurons = %d" % self.nn.inputNeurons.shape[0])
        print("the number of hiddenNeurons = %d" % self.nn.hiddenNeurons.shape[0])
        print("the number of outputNeurons = %d" % self.nn.outputNeurons.shape[0])
        print("learning rate = %.3f" % self.nn.r)
        print("momentum = %.3f" % self.nn.m)
        print("learning criterion = %.3f" % self.nn.learning_criterion)
        print("number of patterns = %d" % self.nn.dataset.shape[0])

        self.totalEpochs = 0
        # according to the dataset split it into training and testing
        if self.nn.dataset.shape[0] == 150:
            np.random.shuffle(self.nn.dataset)
            # make the first 0~99 patters as testing pattern
            self.training_set = self.nn.dataset[:100]
            # make the 100~149 patterns as testing pattern
            self.testing_set = self.nn.dataset[-50:150]
        else:
            self.training_set = self.nn.dataset
            self.testing_set = self.nn.dataset

        self.training_set = self.training_set.astype(float)
        self.testing_set = self.testing_set.astype(float)


    def quit(self):
        exit(1)

    # run 100 epochs on the training
    def teach100Epoch(self):
        count = 0
        print("\ntraining from epoch %d to epoch %d" % (self.totalEpochs, self.totalEpochs + 100))

        while count <= 99:
            self.popErr = self.nn.train(self.training_set)
            self.totalEpochs += 1
            count += 1
        print("epoch = %d, popErr = %.6f" % (self.totalEpochs, self.popErr))

    # train neural network until reaching popErr
    def teachToCriteria(self):
        while True:
            self.popErr = self.nn.train(self.training_set)
            self.totalEpochs += 1
            if self.totalEpochs % 100 == 0:
                print("epoch = %d, popErr = %.6f" % (self.totalEpochs, self.popErr))

            if self.popErr < self.nn.learning_criterion:
                print("Reach learning criteria %.3f, stop training, \nepoch = %d, popErr = %.6f" % (self.nn.learning_criterion, self.totalEpochs, self.popErr))
                break

    # Run 100 test, collect accuracy data
    def run100TestAndCollectData(self):
        self.test_T = 0.0
        self.test_F = 0.0
        for _ in range(100):
            self.test()
        self.test_total = self.test_T + self.test_F
        self.accuracy = self.test_T / self.test_total
        print("the accuracy = %d / %d = %.3f" % (self.test_T, self.test_total,  self.accuracy))

    # keep training until the accuracy is above defined value
    def teachToAccuracy(self):
        while self.accuracy < self.accuracy_criteria:
            self.teach100Epoch()
            self.test_T = 0.0
            self.test_F = 0.0
            for _ in range(100):
                # randomly select one patterns from testing set
                np.random.shuffle(self.testing_set)
                # pattern is a tuple with shape (x,)
                pattern = self.testing_set[0]
                # get input part
                input = pattern[:self.nn.num_of_inputAttr]
                i = self.nn.num_of_outputAttr
                # if pattern is from iris dataset, then j = 7, i = 3
                j = pattern.shape[0]
                teaching_input = pattern[-i:j]

                self.nn.compute_forward(input)

                correct = True
                # For every attribute, if the difference between one pair of corresponding attribute is too big => false
                for i in range(self.nn.num_of_outputAttr):
                    if abs(teaching_input[i] - self.nn.outputNeurons[i]) > self.fit_criteria:
                        correct = False
                        break
                if correct:
                    self.test_T += 1
                else:
                    self.test_F += 1
            self.test_total = self.test_T + self.test_F
            self.accuracy = self.test_T / self.test_total
            print("during testing, the popErr = %.6f, the accuracy = %.3f, continue" % (self.popErr, self.accuracy))
            self.popErr_record.append(self.popErr)
            self.accuracy_record.append(self.accuracy)
            self.epoch_record.append(self.totalEpochs)
        print("Epoch = %d, Accuracy = %.3f" % (self.totalEpochs, self.accuracy))

        fig, ax1 = plt.subplots()
        ax1.set_title("number of hidden units = %d, learning rate = %.3f" % (self.nn.hiddenNeurons.shape[0], self.nn.r))
        ax1.plot(self.epoch_record, self.popErr_record, 'b-')
        ax1.set_xlabel('epochs')
        # Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel('popErr', color='b')
        ax1.tick_params('y', colors='b')

        ax2 = ax1.twinx()
        ax2.plot(self.epoch_record, self.accuracy_record, 'r-')
        ax2.set_ylabel('accuracy rate', color='r')
        ax2.tick_params('y', colors='r')

        fig.tight_layout()
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        plt.savefig("popErr_vs_accuracy" + time_str + ".png")
        print("save popErr_vs_accuracy" + time_str + ".png in the current directory")

    # randomly pick a pattern from testing set to test
    def test(self):
        np.random.shuffle(self.testing_set)
        pattern = self.testing_set[0]
        input = pattern[:self.nn.num_of_inputAttr]
        i = self.nn.num_of_outputAttr
        j = pattern.shape[0]
        teaching_input = pattern[-i:j]

        print("\nthe input is: " + str(input))
        print("the teaching input is: " + str(teaching_input))
        self.nn.compute_forward(input)
        # the transpose and index slice is used to make display format be nice
        print("the output is: " + str(self.nn.outputNeurons.T[0]))

        correct = True
        for i in xrange(self.nn.num_of_outputAttr):
            if abs(teaching_input[i] - self.nn.outputNeurons[i]) > self.fit_criteria:
                correct = False
                print("the differences between corresponding attribute is > 0.45, so decide it is False.")
                break
        if correct:
            print("the differences between corresponding attribute are all <= 0.45, so decide it is True.")
            self.test_T += 1
        else:
            self.test_F += 1

    def checkWeights(self):
        print("\n==weights between input and hidden layer are==")
        print(str(self.nn.W_1))
        print("==bias for hidden units are==")
        print(str(self.nn.B_h))

        print("==weights between hidden layer and output are==")
        print(str(self.nn.W_2))
        print("==bias for output units are==")
        print(str(self.nn.B_o))

if __name__ == '__main__':
    controller = Controller()
    while True:
        controller.show_menu()
        selected = input("your choice => ")
        if selected == 1:
            controller.initialize()
        elif selected == 2:
            controller.teach100Epoch()
        elif selected == 3:
            controller.teachToAccuracy()
        elif selected == 4:
            controller.teachToCriteria()
        elif selected == 5:
            controller.test()
        elif selected == 6:
            controller.checkWeights()
        elif selected == 7:
            controller.run100TestAndCollectData()
        elif selected == 8:
            controller.checkHiddenUnits()
        elif selected == 9:
            controller.checkSetting()
        elif selected == 0:
            controller.quit()
        else:
            print("Invalid input.")
            continue


