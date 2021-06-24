from CvHandler import CvHandler
from InferenceHandler import InferenceHandler
from time import sleep

class EPIDetector():
    def __init__(self, source, person, model, device):
        self.frames = frames = {}
        self.cv = CvHandler(source)
        self.inference = InferenceHandler(person, model, device)
        self.stopped= False

    def start_loop(self, result):
        self.cv.start_capture(self.frames)
        try:
            while not self.stopped:
                if len(self.frames) > 0:
                    result['latest'] = self.inference.process_frame(self.frames['latest'])
                    result['new'] = True
        except KeyboardInterrupt:
            pass
        self.cv.stop_capture()
    
    def stop_loop(self):
        self.stopped = True