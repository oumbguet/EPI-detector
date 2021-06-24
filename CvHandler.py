import cv2
import threading

class CvHandler():
    def __init__(self, source):
        self.src = source
        self.cap = cv2.VideoCapture(self.src)
        self.stopped = False
        self.thr = None
        print("[info] opencv handler initiated")

    def read_cap(self, dest):
        while(self.cap.isOpened() and not self.stopped):
            ret, frame = self.cap.read()
            if ret:
                dest['latest'] = frame
            else:
                print("[info] Catching up with stream")
                self.cap = cv2.VideoCapture(self.src)
        cv2.destroyAllWindows()

    def start_capture(self, destination):
        self.thr = threading.Thread(target=self.read_cap, args=(destination,))
        self.thr.start()

    def stop_capture(self):
        self.stopped = True
        self.thr.join()
        print("[info] Capture thread ended")

    def show_frame(self, frame):
        cv2.imshow('result', frame)
    
    def write_frame(self, frame, path):
        print("[info] writing file " + path)
        cv2.imwrite(path, frame)