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

from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

from tensorflow.python.framework import tensor_util

import cv2
import numpy as np
import time

tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS

def printResult(result, labels):
  boxes = result.outputs['boxes']
  classes = result.outputs['classes']
  scores = result.outputs['scores']

  tt = tensor_util.MakeNdarray(classes)
  # for t in tt:
  #   print(labels[t])

def main(_):

  labels = open("/home/yitao/Documents/fun-project/devicehive-video-analysis/data/yolo2/yolo2.names").read().splitlines()

  host, port = FLAGS.server.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'devicehive_yolo'
  request.model_spec.signature_name = 'predict_images'

  # image_name = "/home/yitao/Documents/TF-Serving-Downloads/dog.jpg"
  image_name = "/home/yitao/Documents/fun-project/devicehive-video-analysis/1.png"

  iteration_list = [15, 1, 10]
  for iteration in iteration_list:
    start = time.time()
    for i in range(iteration):
      data = cv2.imread(image_name)
      data = cv2.resize(data, (608, 608))
      data = data / 255.
      data = np.expand_dims(data, 0)

      print(data.shape)

      request.inputs['input'].CopyFrom(
        tf.contrib.util.make_tensor_proto(data, dtype = np.float32, shape=data.shape))

      result = stub.Predict(request, 10.0)

      printResult(result, labels)
    end = time.time()
    print("It takes %s sec to run %d images for YOLO-devicehive" % (str(end - start), iteration))
  

if __name__ == '__main__':
  tf.app.run()