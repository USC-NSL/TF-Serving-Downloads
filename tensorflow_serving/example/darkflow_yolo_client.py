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

import cv2
import numpy as np

from tensorflow.python.framework import tensor_util

tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS

def resize_input(im):
  imsz = cv2.resize(im, (608, 608)) # hard-coded 608 according to predict.py's log...
  imsz = imsz / 255.
  imsz = imsz[:,:,::-1]
  return imsz

def main(_):
  host, port = FLAGS.server.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  # Send request
  # with open(FLAGS.image, 'rb') as f:
    # See prediction_service.proto for gRPC request/response details.
  # data = f.read()
  # image_name = "/home/yitao/Documents/fun-project/darknet-repo/darkflow/sample_img/sample_person.jpg"
  image_name = "/home/yitao/Documents/TF-Serving-Downloads/cat.jpg"
  im = cv2.imread(image_name)
  # h, w, _ = im.shape
  im = resize_input(im)
  this_inp = np.expand_dims(im, 0)

  print(this_inp.dtype)
  # print(this_inp.shape)
  # print(this_inp)

  # tmp = tf.contrib.util.make_tensor_proto(this_inp, dtype = tf.float64, shape=this_inp.shape)
  # print(tmp.dtype)
  # print(tmp.tensor_shape)

  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'darkflow_yolo'
  request.model_spec.signature_name = 'predict_images'
  request.inputs['input'].CopyFrom(
      tf.contrib.util.make_tensor_proto(this_inp, dtype = np.float32, shape=this_inp.shape))

  # print(request.inputs['input'].tensor_shape)

  result = stub.Predict(request, 10.0)  # 10 secs timeout
  # print(result.outputs['output'])
  tmp = result.outputs['output']
  print(tmp.tensor_shape)

  tt = tensor_util.MakeNdarray(tmp)[0]
  print(tt.shape)
  print(tt[:,0,0])
  print(tt[:,0,1])


if __name__ == '__main__':
  tf.app.run()