from PyQt4.QtCore import *
from PyQt4.QtGui import *
import sys


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




if __name__ == '__main__':
    app = QApplication(sys.argv)

    form = Form()
    form.show()
    app.exec_()