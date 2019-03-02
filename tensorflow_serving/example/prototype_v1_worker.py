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
from tensorflow_serving.apis import olympian_worker_grpc_pb2

import time

import logging
logging.basicConfig()

from concurrent import futures
import grpc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
MAX_MESSAGE_LENGTH = 1024 * 1024 * 64

class OlympianWorker(olympian_worker_grpc_pb2.OlympianWorkerServicer):

  def printRouteTable(self, route_table, machine_name):
    tmp = route_table.split("-")
    for i in range(len(tmp)):
      print("[%s] route info: hop-%s %s" % (machine_name, str(i).zfill(2), tmp[i]))

  def Predict(self, request, context):
    if (request.model_spec.signature_name == "chain_specification"): # gRPC from client
      chain_name = request.model_spec.name
      client_input = tensor_util.MakeNdarray(request.inputs["client_input"])
      print("[Worker] Received request using chain %s with client_input = %s" % (chain_name, str(client_input)))

      route_table = tensor_util.MakeNdarray(request.inputs["route_table"])
      # print("[Worker] %s" % route_table)
      self.printRouteTable(str(route_table), "Worker")

      print(" ")


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
  server.add_insecure_port('localhost:50101')
  server.start()
  try:
    while True:
      time.sleep(_ONE_DAY_IN_SECONDS)
  except KeyboardInterrupt:
    server.stop(0)


if __name__ == '__main__':
  tf.app.run()