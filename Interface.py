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
from  argparse import ArgumentParser, SUPPRESS

def build_argparser():
    parser = ArgumentParser()
    args = parser.add_argument_group('Options')
    args.add_argument('-m', '--model', help="Path to an .xml file, representing a trained person detector SSD model.", required=True, type=str)
    args.add_argument('-c', '--classification', help="Path to an .xml file, representing a trained classification model.", required=True, type=str)
    args.add_argument('-d', '--device', help="CPU, GPU, MYRIAD.", default="CPU", type=str)
    args.add_argument('-i', '--input', help="Input video source.", default="0", type=str)
    args.add_argument('-r', '--ratio', help="Ratio percentage for the video output size.", default=100, type=int)

    return parser

class Window(QWidget):
    def __init__(self):
        QWidget.__init__(self)

        self.args = build_argparser().parse_args()

        self.result = {}
        self.detector = EPIDetector((0 if self.args.input == "0" else self.args.input), self.args.model, self.args.classification, self.args.device)

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
        if (self.args.ratio != 100):
            height = int(img.shape[0] / 100 * self.args.ratio)
            width = int(img.shape[1] / 100 * self.args.ratio)
            dim = (width, height)
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
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
