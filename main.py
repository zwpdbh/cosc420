from PyQt4.QtCore import *
from PyQt4.QtGui import *
import sys
import numpy as np
from NeuralNetwork import NN

'''
Main menu: 
 1 : initialise 
 2 : teach (100 epochs)
 3:  teach (to criteria) 
 4 : test
 5:  show weights
 0 : quit
'''

class Form(QDialog):
    def __init__(self, parent=None):
        super(Form, self).__init__(parent)

        self.leftwg = QWidget()
        self.righwg = QWidget()
        self.totalwg = QWidget()

        # create components for left side
        self.initButton = QPushButton("Initialize")
        self.teachEpochButton = QPushButton("teach 100 epochs")
        self.teachCriteriaButton = QPushButton("teach to criteria")
        self.testButton = QPushButton("test")
        self.showWeightsButton = QPushButton("show weights")
        self.stopButton = QPushButton("stop")

        left_wg_layout = QVBoxLayout()
        left_wg_layout.addWidget(self.initButton)
        left_wg_layout.addWidget(self.teachEpochButton)
        left_wg_layout.addWidget(self.teachCriteriaButton)
        left_wg_layout.addWidget(self.testButton)
        left_wg_layout.addWidget(self.showWeightsButton)
        left_wg_layout.addWidget(self.stopButton)
        self.leftwg.setLayout(left_wg_layout)

        # create component for right side
        self.logwg = QPlainTextEdit(parent)
        self.logwg.setReadOnly(True)

        right_wg_layout = QVBoxLayout()
        right_wg_layout.addWidget(self.logwg)
        self.righwg.setLayout(right_wg_layout)

        # combine left and right widgets
        total_wg_layout = QHBoxLayout()
        total_wg_layout.addWidget(self.leftwg)
        total_wg_layout.addWidget(self.righwg)
        self.totalwg.setLayout(total_wg_layout)

        self.setLayout(total_wg_layout)
        self.setWindowTitle("Neural Network")

        self.connect(self.initButton, SIGNAL("clicked()"), self.initialize)
        self.connect(self.teachEpochButton, SIGNAL("clicked()"), self.teach100Epoch)
        self.connect(self.teachCriteriaButton, SIGNAL("clicked()"), self.teachToCriteria)
        self.connect(self.testButton, SIGNAL("clicked()"), self.test)
        self.connect(self.stopButton, SIGNAL("clicked()"), self.stop)

    def initialize(self):
        # params = np.loadtxt('param.txt')
        # inputs = np.loadtxt('input.txt')
        # teachingInput = np.loadtxt('teaching_input.txt')

        params = np.loadtxt('encoder_param.txt')
        inputs = np.loadtxt('encoder_input.txt')
        teachingInput = np.loadtxt('encoder_teaching_input.txt')

        self.logFile = open('logfile.txt', 'w')

        self.nn = NN(params, inputs, teachingInput)
        self.nn.initializeWeights()

        self.logwg.appendPlainText("Initialization complete, current settings are:")
        self.logwg.appendPlainText("the number of inputNeurons = %d" % self.nn.inputNeurons.shape[0])
        self.logwg.appendPlainText("the number of hiddenNeurons = %d" % self.nn.hiddenNeurons.shape[0])
        self.logwg.appendPlainText("the number of outputNeurons = %d" % self.nn.outputNeurons.shape[0])
        self.logwg.appendPlainText("learning rate = %.3f" % self.nn.r)
        self.logwg.appendPlainText("momentum = %.3f" % self.nn.m)
        self.logwg.appendPlainText("learning criterion = %.3f" % self.nn.learning_criterion)
        self.logwg.appendPlainText("\n")

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

    def stop(self):
        self.logFile.close()

    def teach100Epoch(self):
        print "\ntraining from epoch %d to epoch %d" % (self.totalEpochs, self.totalEpochs + 100)
        # self.logwg.appendPlainText("training from epoch %d to epoch %d" % (self.totalEpochs, self.totalEpochs + 100))
        self.popErr = self.nn.train(self.training_set)
        self.totalEpochs += 100
        print "epoch = %d, popErr = %.6f" % (self.totalEpochs, self.popErr)
        if not self.logFile.close():
            self.logFile.close()
        self.logFile = open('logfile.txt', 'a')
        self.logFile.write("%d %.6f\n" % (self.totalEpochs, self.popErr))
        # self.logwg.appendPlainText("epoch = %d, popErr = %.3f" % (self.totalEpochs, self.popErr))

    def teachToCriteria(self):
        while True:
            self.teach100Epoch()
            if self.popErr < self.nn.learning_criterion:
                # self.logwg.appendPlainText("Reach learning criteria, stop training")
                print "Reach learning criteria, stop training"
                break
        if not self.logFile.close():
            self.logFile.close()

    def test(self):
        np.random.shuffle(self.testing_set)
        pattern = self.testing_set[0]
        input = pattern[:self.nn.num_of_inputAttr]
        i = self.nn.num_of_outputAttr
        j = pattern.shape[0]
        teaching_input = pattern[-i:j]

        self.logwg.appendPlainText("\nthe input is: " + str(input))
        self.logwg.appendPlainText("the teaching input is: " + str(teaching_input))
        self.nn.compute_forward(input)
        # the transpose and index slice is used to make display format be nice
        self.logwg.appendPlainText("the output is: " + str(self.nn.outputNeurons.T[0]))

    def checkWeights(self):
        pass
        # print "W_1 : "
        # print nn.W_1
        #
        # print "W_2 : "
        # print nn.W_2
        #
        # print "B_h: "
        # print nn.B_h
        #
        # print "B_o:"
        # print nn.B_o

if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = Form()
    form.show()
    app.exec_()