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

from tensorflow.python.framework import tensor_util

# from tensorflow_serving.apis import tomtest_pb2
# from tensorflow_serving.apis import tomtest_grpc_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow_serving.apis import olympian_master_grpc_pb2
from tensorflow_serving.apis import olympian_worker_grpc_pb2

import time
import numpy as np

import logging
logging.basicConfig()

from concurrent import futures
import grpc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
MAX_MESSAGE_LENGTH = 1024 * 1024 * 64

tf.app.flags.DEFINE_string('worker', 'localhost:50100',
                           'Olympian worker host:port')
FLAGS = tf.app.flags.FLAGS

# Worker Class
class OlympianWorker(olympian_worker_grpc_pb2.OlympianWorkerServicer):

  def __init__(self):
    self.cstubs = dict()

    # add worker stub
    worker_list = ["localhost:50101", "localhost:50102"]
    for w in worker_list:
      channel = grpc.insecure_channel(w)
      stub = olympian_worker_grpc_pb2.OlympianWorkerStub(channel)
      self.cstubs[w] = stub
    # add master stub
    master_list = ["localhost:50051"]
    for m in master_list:
      channel = grpc.insecure_channel(m)
      stub = olympian_master_grpc_pb2.OlympianMasterStub(channel)
      self.cstubs[m] = stub

    # add istub for internal TF-Serving
    ichannel = implementations.insecure_channel("localhost", 9000)
    self.istub = prediction_service_pb2.beta_create_PredictionService_stub(ichannel)

  def getStubInfo(self, route_table, current_stub):
    tmp = route_table.split("-")
    for i in range(len(tmp)):
      t = tmp[i]
      tt = t.split(":")
      tstub = "%s:%s" % (tt[1], tt[2])
      if (tstub == current_stub):
        current_model = tt[0]
        ttt = tmp[i + 1]
        tttt = ttt.split(":")
        next_stub = "%s:%s" % (tttt[1], tttt[2])
        return current_model, next_stub
    return "Error", "Error"

  def printRouteTable(self, route_table, machine_name):
    tmp = route_table.split("-")
    for i in range(len(tmp)):
      print("[%s][%s] route info: hop-%s %s" % (str(time.time()), machine_name, str(i).zfill(2), tmp[i]))

  # def load_labels(self):
  #   label_file = ("/home/yitao/Documents/fun-project/tensorflow-related/tensorflow-for-poets-2/tf_files/retrained_labels.txt")
  #   label = []
  #   proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  #   for l in proto_as_ascii_lines:
  #     label.append(l.rstrip())
  #   return label

  def Predict(self, request, context):
    if (request.model_spec.signature_name == "chain_specification"): # gRPC from client
      chain_name = request.model_spec.name

      if ("input" in request.inputs):
        request_input = tensor_util.MakeNdarray(request.inputs["input"])
      else:
        request_input = tensor_util.MakeNdarray(request.inputs["normalized_image"])

      print("[%s][Worker] Received request using chain %s with request_input.shape = %s" % (str(time.time()), chain_name, str(request_input.shape)))

      route_table = tensor_util.MakeNdarray(request.inputs["route_table"])
      # self.printRouteTable(str(route_table), "Worker")

      current_model, next_stub = self.getStubInfo(str(route_table), FLAGS.worker)
      print("[%s][Worker] current_model = %s" % (time.time(), current_model))
      print("                        next_stub = %s\n" % (next_stub))



      if (current_model == "exported_mobilenet_v1_1.0_224_preprocess"):
        internal_request = predict_pb2.PredictRequest()
        internal_request.model_spec.name = "exported_mobilenet_v1_1.0_224_preprocess"
        internal_request.model_spec.signature_name = 'predict_images'

        internal_request.inputs['input_image_name'].CopyFrom(
          tf.contrib.util.make_tensor_proto(request_input))

        internal_result = self.istub.Predict(internal_request, 10.0)

        internal_result_value = tensor_util.MakeNdarray(internal_result.outputs["normalized_image"])
        print(internal_result_value.shape)
        # print(internal_result_value[0, 90:95, 205, :])

        next_request = predict_pb2.PredictRequest()
        next_request.model_spec.name = chain_name
        next_request.model_spec.signature_name = "chain_specification"

        next_request.inputs['normalized_image'].CopyFrom(
          tf.contrib.util.make_tensor_proto(internal_result_value, shape=[1, 224, 224, 3]))
        next_request.inputs['route_table'].CopyFrom(
        tf.contrib.util.make_tensor_proto(str(route_table)))

        next_result = self.cstubs[next_stub].Predict(next_request, 10.0)

      elif (current_model == "exported_mobilenet_v1_1.0_224_inference"):
        internal_request = predict_pb2.PredictRequest()
        internal_request.model_spec.name = "exported_mobilenet_v1_1.0_224_inference"
        internal_request.model_spec.signature_name = 'predict_images'

        internal_request.inputs['normalized_image'].CopyFrom(
          tf.contrib.util.make_tensor_proto(request_input, shape=[1, 224, 224, 3]))

        internal_result = self.istub.Predict(internal_request, 10.0)

        internal_result_value = tensor_util.MakeNdarray(internal_result.outputs["scores"])
        print(internal_result_value.shape)

        next_request = predict_pb2.PredictRequest()
        next_request.model_spec.name = chain_name
        next_request.model_spec.signature_name = "chain_specification"

        next_request.inputs['FINAL'].CopyFrom(
          tf.contrib.util.make_tensor_proto(internal_result_value, shape=[1, 5]))

        next_result = self.cstubs[next_stub].Predict(next_request, 10.0)

        # labels = self.load_labels()
        # results = np.squeeze(internal_result_value)
        # top_k = results.argsort()[-5:][::-1]
        # for i in top_k:
        #   print(labels[i], results[i])





      dumbresult = predict_pb2.PredictResponse()
      dumbresult.outputs["message"].CopyFrom(tf.contrib.util.make_tensor_proto("OK"))
      return dumbresult

    else: # Not sure yet...
      print("[Worker] Not sure yet...")
      dumbresult = predict_pb2.PredictResponse()
      dumbresult.outputs["message"].CopyFrom(tf.contrib.util.make_tensor_proto("OK"))
      return dumbresult


def main(_):
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH), 
                                                                    ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                                                                    ('grpc.max_message_length', MAX_MESSAGE_LENGTH)])
  olympian_worker_grpc_pb2.add_OlympianWorkerServicer_to_server(OlympianWorker(), server)
  server.add_insecure_port(FLAGS.worker)
  server.start()
  try:
    while True:
      time.sleep(_ONE_DAY_IN_SECONDS)
  except KeyboardInterrupt:
    server.stop(0)


if __name__ == '__main__':
  tf.app.run()