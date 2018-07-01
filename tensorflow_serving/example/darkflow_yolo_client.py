# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python2.7

"""Send JPEG image to tensorflow_model_server loaded with inception model.
"""

from __future__ import print_function

# This is a placeholder for a Google-internal import.

import grpc
from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

import cv2
import numpy as np
import time

from tensorflow.python.framework import tensor_util

from darkflow.cython_utils.cy_yolo2_findboxes import box_constructor

tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS

def resize_input(im):
  imsz = cv2.resize(im, (608, 608)) # hard-coded 608 according to predict.py's log...
  imsz = imsz / 255.
  imsz = imsz[:,:,::-1]
  return imsz

def process_box(b, h, w, threshold, meta):
  max_indx = np.argmax(b.probs)
  max_prob = b.probs[max_indx]
  label = meta['labels'][max_indx]
  if max_prob > threshold:
    left  = int ((b.x - b.w/2.) * w)
    right = int ((b.x + b.w/2.) * w)
    top   = int ((b.y - b.h/2.) * h)
    bot   = int ((b.y + b.h/2.) * h)
    if left  < 0    :  left = 0
    if right > w - 1: right = w - 1
    if top   < 0    :   top = 0
    if bot   > h - 1:   bot = h - 1
    mess = '{}'.format(label)
    return (left, right, top, bot, mess, max_indx, max_prob)
  return None

def main(_):
  host, port = FLAGS.server.split(':')
  
  channel = grpc.insecure_channel("%s:%s" % (host, port))
  stub = prediction_service_pb2.PredictionServiceStub(channel)

  # channel = implementations.insecure_channel(host, int(port))
  # stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'darkflow_yolo'
  request.model_spec.signature_name = 'predict_images'

  # Send request
  # with open(FLAGS.image, 'rb') as f:
    # See prediction_service.proto for gRPC request/response details.
  # data = f.read()
  # image_name = "/home/yitao/Documents/fun-project/darknet-repo/darkflow/sample_img/sample_person.jpg"
  # image_name = "/home/yitao/Documents/TF-Serving-Downloads/dog.jpg"
  image_name = "/home/yitao/Documents/TF-Serving-Downloads/cat.jpg"
  
  iteration_list = [5]
  # iteration_list = [15, 1, 10, 20, 40, 80, 160, 320]
  for iteration in iteration_list:
    start = time.time()
    for i in range(iteration):
      im = cv2.imread(image_name)

      print("[%s] start pre-processing" % str(time.time()))

      h, w, _ = im.shape
      im = resize_input(im)
      this_inp = np.expand_dims(im, 0)

      # print(this_inp.dtype)
      # print(this_inp.shape)
      # print(this_inp)

      # tmp = tf.contrib.util.make_tensor_proto(this_inp, dtype = tf.float64, shape=this_inp.shape)
      # print(tmp.dtype)
      # print(tmp.tensor_shape)

      request.inputs['input'].CopyFrom(
          tf.contrib.util.make_tensor_proto(this_inp, dtype = np.float32, shape=this_inp.shape))

      # print(request.inputs['input'].tensor_shape)

      print("[%s] start processing" % str(time.time()))

      result = stub.Predict(request, 10.0)  # 10 secs timeout

      print("[%s] start post-processing" % str(time.time()))

      # print(result.outputs['output'])
      tmp = result.outputs['output']
      # print(tmp.tensor_shape)

      tt = tensor_util.MakeNdarray(tmp)[0]
      # print(tt.shape)
      # print(tt[:,0,0])
      # print(tt[:,0,1])

      meta = {'labels': ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'], u'jitter': 0.3, u'anchors': [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828], u'random': 1, 'colors': [(254, 254, 254), (254, 254, 127), (254, 254, 0), (254, 254, -127), (254, 254, -254), (254, 127, 254), (254, 127, 127), (254, 127, 0), (254, 127, -127), (254, 127, -254), (254, 0, 254), (254, 0, 127), (254, 0, 0), (254, 0, -127), (254, 0, -254), (254, -127, 254), (254, -127, 127), (254, -127, 0), (254, -127, -127), (254, -127, -254), (254, -254, 254), (254, -254, 127), (254, -254, 0), (254, -254, -127), (254, -254, -254), (127, 254, 254), (127, 254, 127), (127, 254, 0), (127, 254, -127), (127, 254, -254), (127, 127, 254), (127, 127, 127), (127, 127, 0), (127, 127, -127), (127, 127, -254), (127, 0, 254), (127, 0, 127), (127, 0, 0), (127, 0, -127), (127, 0, -254), (127, -127, 254), (127, -127, 127), (127, -127, 0), (127, -127, -127), (127, -127, -254), (127, -254, 254), (127, -254, 127), (127, -254, 0), (127, -254, -127), (127, -254, -254), (0, 254, 254), (0, 254, 127), (0, 254, 0), (0, 254, -127), (0, 254, -254), (0, 127, 254), (0, 127, 127), (0, 127, 0), (0, 127, -127), (0, 127, -254), (0, 0, 254), (0, 0, 127), (0, 0, 0), (0, 0, -127), (0, 0, -254), (0, -127, 254), (0, -127, 127), (0, -127, 0), (0, -127, -127), (0, -127, -254), (0, -254, 254), (0, -254, 127), (0, -254, 0), (0, -254, -127), (0, -254, -254), (-127, 254, 254), (-127, 254, 127), (-127, 254, 0), (-127, 254, -127), (-127, 254, -254)], u'num': 5, u'thresh': 0.1, 'inp_size': [608, 608, 3], u'bias_match': 1, 'out_size': [19, 19, 425], 'model': '/home/yitao/Documents/fun-project/darknet-repo/darkflow/cfg/yolo.cfg', u'absolute': 1, 'name': 'yolo', u'coord_scale': 1, u'rescore': 1, u'class_scale': 1, u'noobject_scale': 1, u'object_scale': 5, u'classes': 80, u'coords': 4, u'softmax': 1, 'net': {u'hue': 0.1, u'saturation': 1.5, u'angle': 0, u'decay': 0.0005, u'learning_rate': 0.001, u'scales': u'.1,.1', u'batch': 1, u'height': 608, u'channels': 3, u'width': 608, u'subdivisions': 1, u'burn_in': 1000, u'policy': u'steps', u'max_batches': 500200, u'steps': u'400000,450000', 'type': u'[net]', u'momentum': 0.9, u'exposure': 1.5}, 'type': u'[region]'}

      boxes = list()
      boxes = box_constructor(meta, tt)

      # for box in boxes:
        # print("(%s, %s, %s, %s) with class_num = %s, probs = %s" % (str(box.x), str(box.y), str(box.w), str(box.h), str(box.class_num), str(box.probs)))

      # set threshold
      threshold = 0.1

      boxesInfo = list()
      for box in boxes:
        tmpBox = process_box(box, h, w, threshold, meta)
        if tmpBox is None:
          continue
        boxesInfo.append({
          "label": tmpBox[4],
          "confidence": tmpBox[6],
          "topleft": {
            "x": tmpBox[0],
            "y": tmpBox[2]},
          "bottomright": {
            "x": tmpBox[1],
            "y": tmpBox[3]}
          })

      print("[%s] finished post-processing" % str(time.time()))

      # for res in boxesInfo:
      #   print("%s (confidence: %s)" % (res['label'], str(res['confidence'])))
      #   print("    topleft(%s, %s), botright(%s, %s)" % 
      #       (res['topleft']['x'], res['topleft']['y'], 
      #       res['bottomright']['x'], res['bottomright']['y']))

    end = time.time()

    print("It takes %s sec to run %d images for YOLO" % (str(end - start), iteration))

if __name__ == '__main__':
  tf.app.run()