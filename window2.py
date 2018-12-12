import sys
from PyQt5 import QtWidgets
# from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QAction, QPushButton, QLabel, QAbstractButton, QPainter
# from PyQt5.QtGui import QIcon, QPixmap
# from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QMainWindow


class PicButton(QAbstractButton):
    def __init__(self, pixmap, parent=None):
        super(PicButton, self).__init__(parent)
        self.pixmap = pixmap

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(event.rect(), self.pixmap)

    def sizeHint(self):
        return self.pixmap.size()



class App2(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'window2'
        self.left = 200
        self.top = 80
        self.width = 900
        self.height = 600
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        # imageStart = QPixmap('image/start.png')
        # iconStart = QIcon(imageStart)
        # buttonStart = QPushButton(iconStart,"",self)
        # buttonStart.move(750,400)
        # buttonStart.clicked.connect(self.click)
        # label = QLabel(self)
        # label.setPixmap(imageStart)

        # Set window background color
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QColor(145, 245, 104))
        self.setPalette(p)

        btn = PicButton(QPixmap('E:\pypt5\end.png'), self)
        btn.move(800, 0)
        btn.resize(100, 100)
        btn.clicked.connect(self.click)
        self.show()

    def createHorizontalLayout(self):
        self.horizontalGroupBox = QGroupBox("What is your favorite color?")
        layout = QHBoxLayout()

        buttonBlue = QPushButton('Blue', self)
        buttonBlue.clicked.connect(self.on_click)
        layout.addWidget(buttonBlue)

        buttonRed = QPushButton('Red', self)
        buttonRed.clicked.connect(self.click)
        layout.addWidget(buttonRed)

        buttonGreen = QPushButton('Green', self)
        buttonGreen.clicked.connect(self.click)
        layout.addWidget(buttonGreen)

        self.horizontalGroupBox.setLayout(layout)

    @pyqtSlot()
    def click(self):
        self.run = False
        print('PyQt5 button click')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App2()
    sys.exit(app.exec_())