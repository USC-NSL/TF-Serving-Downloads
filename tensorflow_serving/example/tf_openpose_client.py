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
import time

tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS


def main(_):
  host, port = FLAGS.server.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'tf_openpose'
  request.model_spec.signature_name = 'predict_images'

  image_name = "/home/yitao/Documents/fun-project/tf-pose-estimation/images/p1.jpg"
  my_upsample_size = [116, 108]

  iteration_list = [15, 1, 10]
  for iteration in iteration_list:
    start = time.time()
    for i in range(iteration):

      print("[%s] start pre-processing" % str(time.time()))

      data = cv2.imread(image_name, cv2.IMREAD_COLOR)
      data = np.expand_dims(data, 0)

      # print(data.shape)

      request.inputs['tensor_image'].CopyFrom(
        tf.contrib.util.make_tensor_proto(data, dtype = np.float32, shape=data.shape))
      request.inputs['upsample_size'].CopyFrom(
        tf.contrib.util.make_tensor_proto(my_upsample_size))

      print("[%s] start processing" % str(time.time()))

      result = stub.Predict(request, 10.0)

      print("[%s] finish processing" % str(time.time()))

      # tensor_peaks = result.outputs["tensor_peaks"]
      # tensor_heatMat_up = result.outputs["tensor_heatMat_up"]
      # tensor_pafMat_up = result.outputs["tensor_pafMat_up"]

      # print(tensor_peaks.tensor_shape.dim)
      # print(tensor_heatMat_up.tensor_shape.dim)
      # print(tensor_pafMat_up.tensor_shape.dim)

    end = time.time()
    print("It takes %s sec to run %d images for tf-openpose" % (str(end - start), iteration))




if __name__ == '__main__':
  tf.app.run()
