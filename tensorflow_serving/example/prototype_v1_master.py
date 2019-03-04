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

# Master Class
class OlympianMaster(olympian_master_grpc_pb2.OlympianMasterServicer):

  def __init__(self):
    self.cstubs = dict()

    # add worker stub
    # worker_list = ["localhost:50101", "localhost:50102"]
    worker_list = ["192.168.1.125:50101", "192.168.1.102:50102"]
    for w in worker_list:
      channel = grpc.insecure_channel(w)
      stub = olympian_worker_grpc_pb2.OlympianWorkerStub(channel)
      self.cstubs[w] = stub
    # add master stub
    # master_list = ["localhost:50051"]
    master_list = ["192.168.1.102:50051"]
    for m in master_list:
      channel = grpc.insecure_channel(m)
      stub = olympian_master_grpc_pb2.OlympianMasterStub(channel)
      self.cstubs[m] = stub

  def getRouteTable(self, chain_name):
    if (chain_name == "chain_mobilenet"):
      # return "exported_mobilenet_v1_1.0_224_preprocess:localhost:50101-exported_mobilenet_v1_1.0_224_inference:localhost:50102-FINAL:localhost:50051"
      return "exported_mobilenet_v1_1.0_224_preprocess:192.168.1.125:50101-exported_mobilenet_v1_1.0_224_inference:192.168.1.102:50102-FINAL:192.168.1.102:50051"
    else:
      return "Not implemented yet..."

  def getNextStub(self, route_table):
    tmp = route_table.split("-")[0].split(":")
    next_stub = "%s:%s" % (tmp[1], tmp[2])
    return next_stub

  def printRouteTable(self, route_table, machine_name):
    tmp = route_table.split("-")
    for i in range(len(tmp)):
      print("[%s][%s] route info: hop-%s %s" % (str(time.time()), machine_name, str(i).zfill(2), tmp[i]))

  def load_labels(self):
    label_file = ("/home/yitao/Documents/fun-project/tensorflow-related/tensorflow-for-poets-2/tf_files/retrained_labels.txt")
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
      label.append(l.rstrip())
    return label

  def Predict(self, request, context):
    # if (request.model_spec.signature_name == "chain_specification"): # gRPC from client
    if ("FINAL" not in request.inputs): # gRPC from client
      chain_name = request.model_spec.name
      client_input = tensor_util.MakeNdarray(request.inputs["client_input"])
      print("[%s][Master] Received request using chain %s with client_input = %s" % (str(time.time()), chain_name, str(client_input)))

      route_table = self.getRouteTable(chain_name)
      # self.printRouteTable(str(route_table), "Worker")

      next_stub = self.getNextStub(route_table)      
      print("[%s][Master] next stub is %s\n\n" % (str(time.time()), next_stub))

      newrequest = predict_pb2.PredictRequest()
      newrequest.model_spec.name = chain_name
      newrequest.model_spec.signature_name = "chain_specification"
      newrequest.inputs["input"].CopyFrom(
        tf.contrib.util.make_tensor_proto(client_input))
      newrequest.inputs["route_table"].CopyFrom(
        tf.contrib.util.make_tensor_proto(route_table))

      result = self.cstubs[next_stub].Predict(newrequest, 10.0)

      dumbresult = predict_pb2.PredictResponse()
      dumbresult.outputs["message"].CopyFrom(tf.contrib.util.make_tensor_proto("OK"))
      return dumbresult

    else: # gRPC from worker
      # print("[Master] Not implemented yet...")
      final_result_value = tensor_util.MakeNdarray(request.inputs["FINAL"])
      print(final_result_value.shape)

      # Mobilenet specific
      if (request.model_spec.name == "chain_mobilenet"):
        labels = self.load_labels()
        results = np.squeeze(final_result_value)
        top_k = results.argsort()[-5:][::-1]
        for i in top_k:
          print(labels[i], results[i])



      dumbresult = predict_pb2.PredictResponse()
      dumbresult.outputs["message"].CopyFrom(tf.contrib.util.make_tensor_proto("OK"))
      return dumbresult


def main(_):
  

  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH), 
                                                                    ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                                                                    ('grpc.max_message_length', MAX_MESSAGE_LENGTH)])
  olympian_master_grpc_pb2.add_OlympianMasterServicer_to_server(OlympianMaster(), server)
  server.add_insecure_port('[::]:50051')
  server.start()
  try:
    while True:
      time.sleep(_ONE_DAY_IN_SECONDS)
  except KeyboardInterrupt:
    server.stop(0)


if __name__ == '__main__':
  tf.app.run()