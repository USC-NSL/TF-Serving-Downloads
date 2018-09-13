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

import scipy.misc
import numpy as np

tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS


def main(_):
  host, port = FLAGS.server.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  # # Send request
  # with open(FLAGS.image, 'rb') as f:
  #   # See prediction_service.proto for gRPC request/response details.
  #   data = f.read()
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'nvidia_autopilot'
  request.model_spec.signature_name = 'predict_images'
  #   request.inputs['images'].CopyFrom(
  #       tf.contrib.util.make_tensor_proto(data, shape=[1]))
  #   result = stub.Predict(request, 10.0)  # 10 secs timeout
  #   print(result)

  full_image = scipy.misc.imread("/home/yitao/Documents/fun-project/tensorflow-related/Autopilot-TensorFlow/driving_dataset/" + str(200) + ".jpg", mode="RGB")
  image = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0
  keep_prob = 1.0

  request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image, dtype = np.float32, shape=[1, 66, 200, 3]))
  request.inputs['keep_prob'].CopyFrom(
        tf.contrib.util.make_tensor_proto(keep_prob))

  result = stub.Predict(request, 10.0)

  degrees = float(result.outputs["scores"].float_val[0]) * 180.0 / scipy.pi

  print(degrees)

if __name__ == '__main__':
  tf.app.run()