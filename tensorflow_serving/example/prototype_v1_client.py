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
from tensorflow_serving.apis import olympian_master_grpc_pb2

from tensorflow.python.framework import tensor_util

tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS


def main(_):

  file_name = "/home/yitao/Documents/fun-project/tensorflow-related/tensorflow-for-poets-2/tf_files/flower_photos/daisy/21652746_cc379e0eea_m.jpg"

  channel = grpc.insecure_channel("localhost:50051")
  stub = olympian_master_grpc_pb2.OlympianMasterStub(channel)

  request = predict_pb2.PredictRequest()
  request.model_spec.name = "chain_mobilenet"
  request.model_spec.signature_name = "chain_specification"
  request.inputs["client_input"].CopyFrom(
    tf.contrib.util.make_tensor_proto(file_name))

  result = stub.Predict(request, 10.0)
  message = tensor_util.MakeNdarray(result.outputs["message"])
  print(message)


if __name__ == '__main__':
  tf.app.run()