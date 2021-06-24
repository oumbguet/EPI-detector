from __future__ import print_function
import sys
import os
import numpy as np
import logging as log
from openvino.inference_engine import IECore

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import ngraph as ng

import time
import threading
import cv2

from PIL import Image

class InferenceHandler():
    def __init__(self, person_model, classification_model, device):
        self.model = person_model
        self.classification_model = classification_model
        self.device = device
        self.classification = None
        self.ie = None
        self.net = None
        self.exec_net = None
        self.out_blob = None
        self.n, self.c, self.h, self.w = None, None, None, None


        # KERAS
        self.class_net = None
        self.class_exec_net = None


        self.init_models()



    def init_models(self):
        self.ie = IECore()
        self.net = self.ie.read_network(model=self.model)

        # KERAS
        self.class_net = self.ie.read_network(model=self.classification_model)
        # self.classification = keras.models.load_model(self.classification_model)
        self.class_exec_net = self.ie.load_network(network=self.class_net, device_name=self.device)

        self.exec_net = self.ie.load_network(network=self.net, device_name=self.device)

        for input_key in self.net.input_info:
            if len(self.net.input_info[input_key].input_data.layout) == 4:
                self.n, self.c, self.h, self.w = self.net.input_info[input_key].input_data.shape
    
        print("[info] Preparing input blobs")
        self.out_blob = next(iter(self.net.outputs))

        print('[info] Preparing output blobs')
        output_name, output_info = "", None
        func = ng.function_from_cnn(self.net)
        if func:
            ops = func.get_ordered_ops()
            for op in ops:
                if op.friendly_name in self.net.outputs and op.get_type_name() == "DetectionOutput":
                    output_name = op.friendly_name
                    output_info = self.net.outputs[output_name]
                    break
        else:
            output_name = list(self.net.outputs.keys())[0]
            output_info = self.net.outputs[output_name]

        if output_name == "":
            log.error("Can't find a DetectionOutput layer in the topology")
        output_dims = output_info.shape
        if len(output_dims) != 4:
            log.error("Incorrect output dimensions for SSD model")
        max_proposal_count, object_size = output_dims[2], output_dims[3]
        if object_size != 7:
            log.error("Output item should have 7 as a last dimension")
        output_info.precision = "FP32"

    def prediction(self, net, model, img, preds, box):
        start_time = time.time()
        rs_img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        rs_img = rs_img[:,:,::-1].transpose(2,0,1)
        rs_img = np.ascontiguousarray(rs_img)

        input_blob = next(iter(net.inputs))

        res = model.infer(inputs={input_blob: [rs_img]})
        # print(res)

        preds.append(("jaune" if res['reid_embedding'][0][0] < res['reid_embedding'][0][1] else "autre", box))
        end_time = time.time()
        # print("Prediction in " + str(end_time - start_time) + " seconds")

    def process_frame(self, frame):
        start_time = time.time()
        image = frame
        tmp_image = image
        images = np.ndarray(shape=(self.n, self.c, self.h, self.w))
        images_hw = []
        if type(image) == None:
            return
        ih, iw = image.shape[:-1]
        images_hw.append((ih, iw))
        if (ih, iw) != (self.h, self.w):image = cv2.resize(image, (self.w, self.h))
        image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        images[0] = image
        data = {}
        data['data'] = images
        res = self.exec_net.infer(inputs=data)
        res = res[self.out_blob]
        boxes, classes = {}, {}
        data = res[0][0]
        for number, proposal in enumerate(data):
            if proposal[2] > 0:
                imid = np.int(proposal[0])
                ih, iw = images_hw[imid]
                label = np.int(proposal[1])
                confidence = proposal[2]
                xmin = np.int(iw * proposal[3])
                ymin = np.int(ih * proposal[4])
                xmax = np.int(iw * proposal[5])
                ymax = np.int(ih * proposal[6])
                if proposal[2] > 0.75:
                    if not imid in boxes.keys():
                        boxes[imid] = []
                    boxes[imid].append([xmin, ymin, xmax, ymax])
                    if not imid in classes.keys():
                        classes[imid] = []
                    classes[imid].append(label)
        persons = []
        for imid in classes:
            for box in boxes[imid]:
                persons.append((tmp_image[box[1]:box[3], box[0]:box[2]], box))
        predictions = []
        threads = []
        for p in persons:
            person = p[0]
            box = p[1]
            if person.size == 0:
                continue
            threads.append(threading.Thread(target=self.prediction, args=(self.class_exec_net, self.class_exec_net, person, predictions, box)))
            threads[-1].start()
        for t in threads:
            t.join()
        for pred, box in predictions:
            if (pred == 'jaune'):
                tmp_image = cv2.rectangle(tmp_image, (box[0], box[1]), (box[2], box[3]), (0,255,0), 3)
            else:
                tmp_image = cv2.rectangle(tmp_image, (box[0], box[1]), (box[2], box[3]), (0,0,255), 3)
        end_time = time.time()
        print("FPS: " + str(1 / (end_time - start_time)) + "FPS")
        return tmp_image
        