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
        self.pauseButton = QPushButton("pause training")

        left_wg_layout = QVBoxLayout()
        left_wg_layout.addWidget(self.initButton)
        left_wg_layout.addWidget(self.teachEpochButton)
        left_wg_layout.addWidget(self.teachCriteriaButton)
        left_wg_layout.addWidget(self.testButton)
        left_wg_layout.addWidget(self.showWeightsButton)
        left_wg_layout.addWidget(self.pauseButton)
        self.leftwg.setLayout(left_wg_layout)

        # create component for right side
        self.browser = QTextBrowser()
        right_wg_layout = QVBoxLayout()
        right_wg_layout.addWidget(self.browser)
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

    def initialize(self):
        params = np.loadtxt('param.txt')
        inputs = np.loadtxt('input.txt')
        teachingInput = np.loadtxt('teaching_input.txt')
        self.nn = NN(params, inputs, teachingInput)
        self.nn.initializeWeights()

        self.browser.append("Initialization complete, current settings are:")
        self.browser.append("the number of inputNeurons = %d" % self.nn.inputNeurons.shape[0])
        self.browser.append("the number of hiddenNeurons = %d" % self.nn.hiddenNeurons.shape[0])
        self.browser.append("the number of outputNeurons = %d" % self.nn.outputNeurons.shape[0])
        self.browser.append("learning rate = %.3f" % self.nn.r)
        self.browser.append("momentum = %.3f" % self.nn.m)
        self.browser.append("learning criterion = %.3f" % self.nn.learning_criterion)

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


    def teach100Epoch(self):
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = Form()
    form.show()
    app.exec_()