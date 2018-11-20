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
# from __future__ import division, print_function, absolute_import

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS


# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
# from tensorflow_serving.example import mnist_input_data
# mnist = mnist_input_data.read_data_sets("/tmp/data/", one_hot=True)

def main(_):
  host, port = FLAGS.server.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  
  x = mnist.test.images[0:1000]
  y = mnist.test.labels[0:1000]
  keep_prob = 1.0

  # print(x.shape)
  # print(y.shape)

  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'tom_cnn'
  request.model_spec.signature_name = 'prediction'
  request.inputs['input'].CopyFrom(
        tf.contrib.util.make_tensor_proto(x, shape=x.shape))
  request.inputs['keep_prob'].CopyFrom(
        tf.contrib.util.make_tensor_proto(keep_prob))

  for i in range(13):
    result = stub.Predict(request, 10.0)
  
    # print(result.outputs["output"].float_val)
    # print("y = %s" % str(y))

if __name__ == '__main__':
  tf.app.run()