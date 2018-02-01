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

import time
import threading
import sys

tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS

import os.path as osp
import numpy as np

def myFuncWarmUp(stub, i):
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'caffe_resnet152'
  request.model_spec.signature_name = 'predict_images'

  batchSize = 100
  durationSum = 0.0
  runNum = 13

  for k in range(runNum):
    image_data = []
    start = time.time()
    for j in range(batchSize):
      image = "/home/yitao/Downloads/inception-input/%s/dog-%s.jpg" % (str(i % 100).zfill(3), str(j).zfill(3))
      with open(image, 'rb') as f:
        image_data.append(f.read())

    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image_data, shape=[len(image_data)]))

    result = stub.Predict(request, 60.0)  # 10 secs timeout
    # print(result)
    end = time.time()
    duration = (end - start)
    print("it takes %s sec" % str(duration))
    if (k != 0 and k != 3 and k != 8):
      durationSum += duration
    # sys.stdout.write('.')
    # sys.stdout.flush()

  print("[Warm up] on average, it takes %s sec to run a batch of %d images over %d runs" % (str(durationSum / (runNum - 3)), batchSize, (runNum - 3)))

def myFuncParallel(stub, i):
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'caffe_resnet152'
  request.model_spec.signature_name = 'predict_images'

  batchSize = 100
  durationSum = 0.0
  runNum = 1

  for k in range(runNum):
    image_data = []
    start = time.time()
    for j in range(batchSize):
      image = "/home/yitao/Downloads/inception-input/%s/dog-%s.jpg" % (str(i % 100).zfill(3), str(j).zfill(3))
      with open(image, 'rb') as f:
        image_data.append(f.read())

    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image_data, shape=[len(image_data)]))

    result = stub.Predict(request, 60.0)  # 10 secs timeout
    # print(result)
    end = time.time()
    duration = (end - start)
    print("[thread-%d] it takes %s sec" % (i, str(duration)))
    # if (k != 0 and k != 3 and k != 8):
    if True:
      durationSum += duration
    # sys.stdout.write('.')
    # sys.stdout.flush()

  print("[Parallel-thread-%d] on average, it takes %s sec to run a batch of %d images over %d runs" % (i, str(durationSum / runNum), batchSize, runNum))

def main(_):
  # start = time.time()

  host, port = FLAGS.server.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

  # run Inception job
  myFuncWarmUp(stub, 0)

  num_tests = 60
  tPool = []
  for i in range(num_tests):
    tPool.append(threading.Thread(target = myFuncParallel, args = (stub, i)))

  start = time.time()
  for i in range(num_tests):
    t = tPool[i]
    t.start()
    # time.sleep(2.0)

  for i in range(num_tests):
    t = tPool[i]
    t.join()

  end = time.time()

  print('\nFinished!')
  print('[Parallel] The total running time to run %d concurrent jobs is %s' % (num_tests, str(end - start)))


if __name__ == '__main__':
  tf.app.run()
