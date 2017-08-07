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

def myFunc(stub, i):
  input_file = "inception-input/dog-%s.jpg" % str(i).zfill(3)
  # print(input_file)
  with open(input_file, 'rb') as f:
    # See prediction_service.proto for gRPC request/response details.
    data = f.read()
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'inception'
    request.model_spec.signature_name = 'predict_images'
    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(data, shape=[1]))
    
    result = stub.Predict(request, 60.0)
    sys.stdout.write('.')
    sys.stdout.flush()

def main(_):
  host, port = FLAGS.server.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  # Send request
  # with open(FLAGS.image, 'rb') as f:
  #   # See prediction_service.proto for gRPC request/response details.
  #   data = f.read()
  #   request = predict_pb2.PredictRequest()
  #   request.model_spec.name = 'inception'
  #   request.model_spec.signature_name = 'predict_images'
  #   request.inputs['images'].CopyFrom(
  #       tf.contrib.util.make_tensor_proto(data, shape=[1]))

    # minor changes to the original inception_client.py
    # added a loop to test multiple images w/o batching
    # printed meta-data including the running time
    # print(time.time())
    # for i in range(10):
      # start = time.time()
      # result = stub.Predict(request, 60.0)  # 10 secs timeout
      # end = time.time()

      # sys.stdout.write('.')
      # sys.stdout.flush()

      # print("[%s, %s] = %s" % (str(start), str(end), str(end - start)))
      # print(result)

  num_tests = 320
  tPool = []
  for i in range(num_tests):
    tPool.append(threading.Thread(target = myFunc, args = (stub, i)))

  for i in range(num_tests):
    t = tPool[i]
    t.start()
    # time.sleep(2.0)

  for i in range(num_tests):
    t = tPool[i]
    t.join()


  print('\nFinished!')

if __name__ == '__main__':
  tf.app.run()
