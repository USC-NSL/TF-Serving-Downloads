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

tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS

import numpy as np
import time


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


def main(_):
  host, port = FLAGS.server.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

  file_name = "/home/yitao/Documents/fun-project/tensorflow-related/tensorflow-for-poets-2/tf_files/flower_photos/daisy/21652746_cc379e0eea_m.jpg"

  request_preprocess = predict_pb2.PredictRequest()
  request_preprocess.model_spec.name = 'exported_mobilenet_v1_1.0_224_preprocess'
  request_preprocess.model_spec.signature_name = 'predict_images'

  request_preprocess.inputs['input_image_name'].CopyFrom(
    tf.contrib.util.make_tensor_proto(file_name))

  result_preprocess = stub.Predict(request_preprocess, 10.0)
  # print(result.shape)
  result_preprocess_value = tensor_util.MakeNdarray(result_preprocess.outputs['normalized_image'])
  print(result_preprocess_value.shape)
  print(result_preprocess_value[0, 90:95, 205, :])


  # ===========================================================================================================

  request_inference = predict_pb2.PredictRequest()
  request_inference.model_spec.name = 'exported_mobilenet_v1_1.0_224_inference'
  request_inference.model_spec.signature_name = 'predict_images'

  batch_size = 128
  
  iteration_list = [1, 10, 20]
  for iteration in iteration_list:
    start = time.time()
    for i in range(iteration):

      new_input = np.concatenate([result_preprocess_value] * batch_size, axis = 0)

      request_inference.inputs['normalized_image'].CopyFrom(
        tf.contrib.util.make_tensor_proto(new_input, shape=[batch_size, 224, 224, 3]))

      result_inference = stub.Predict(request_inference, 10.0)

      # result_inference_value = tensor_util.MakeNdarray(result_inference.outputs['scores'])
      # print(result_inference_value.shape)

    end = time.time()
    print("It takes %s sec to run %d mobilenet jobs with batch size of %d" % (str(end - start), iteration, batch_size))




  # results = np.squeeze(result_inference_value)

  # top_k = results.argsort()[-5:][::-1]
  # labels = load_labels("/home/yitao/Documents/fun-project/tensorflow-related/tensorflow-for-poets-2/tf_files/retrained_labels.txt")

  # for i in top_k:
  #   print(labels[i], results[i])

if __name__ == '__main__':
  tf.app.run()