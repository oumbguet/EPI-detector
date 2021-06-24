from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout
from PyQt5.QtCore import QThread, QThreadPool, QRunnable
from PyQt5.QtGui import QImage, QPixmap
import sys
import numpy as np
import threading
import cv2
import time
from Detector import EPIDetector

class Window(QWidget):
    def __init__(self):
        QWidget.__init__(self)

        self.result = {}
        self.detector = EPIDetector(0, ".\person_model\person-detection-retail-0013.xml", ".\classification\main_model.xml", "CPU")

        self.frame = QLabel()
        self.stopped = {'bool': False}

        layout = QVBoxLayout()
        layout.addWidget(self.frame)
        self.setLayout(layout)

        self.thr = threading.Thread(target=self.detector.start_loop, args=(self.result,))
        self.thr.start()

        self.thr_update = threading.Thread(target=self.update_frame, args=(self.stopped,))
        self.thr_update.start()

    def update_frame(self, stopped):
        while not stopped['bool']:
            if len(self.result) > 0 and self.result['new']:
                self.change_img(self.result['latest'])
                self.result['new'] = False

    def change_img(self, img_inv):
        img = cv2.cvtColor(img_inv, cv2.COLOR_BGR2RGB)
        height, width, channel = img.shape
        bpl = 3 * width
        qImg = QImage(img.data, width, height, bpl, QImage.Format_RGB888)
        self.frame.setPixmap(QPixmap(qImg))

    def closeEvent(self, a0):
        self.detector.stop_loop()
        self.stopped['bool'] = True
        return super().closeEvent(a0)

if __name__ == '__main__':
    app = QApplication([])
    window = Window()
    window.show()
    app.exec_()
