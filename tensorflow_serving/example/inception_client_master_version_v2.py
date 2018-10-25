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

import time

# tf.app.flags.DEFINE_string('server', 'localhost:9000',
#                            'PredictionService host:port')
# tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
# FLAGS = tf.app.flags.FLAGS


# def main(_):
#   host, port = FLAGS.server.split(':')
#   channel = implementations.insecure_channel(host, int(port))
#   stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
#   # Send request
#   with open(FLAGS.image, 'rb') as f:
#     # See prediction_service.proto for gRPC request/response details.
#     data = f.read()
#     request = predict_pb2.PredictRequest()
#     request.model_spec.name = 'inception'
#     request.model_spec.signature_name = 'predict_images'
#     request.inputs['images'].CopyFrom(
#         tf.contrib.util.make_tensor_proto(data, shape=[1]))
#     result = stub.Predict(request, 10.0)  # 10 secs timeout
#     print(result)

from concurrent import futures
import grpc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
MAX_MESSAGE_LENGTH = 1024 * 1024 * 64

def GetTensorShapeList(ts_shape):
  result = []
  for s in ts_shape.dim:
    result.append(int(s.size))

  return result

def GetNewShapeList(request_shape_list):
  new_request_shape_list = request_shape_list
  new_request_shape_list[0] = request_shape_list[0] / 2
  return new_request_shape_list

class OlympianMaster(olympian_master_grpc_pb2.OlympianMasterServicer):
  # def callServer(input_path):
  #   # host, port = FLAGS.server.split(':')

  def Predict(self, request, context):
    request_shape_list = GetTensorShapeList(request.inputs['images'].tensor_shape)
    print("Master receved request for model %s with shape of %s" % (request.model_spec.name, request_shape_list))

    host = "localhost"
    port = "9000"
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    # newrequest = request

    new_request_shape_list = GetNewShapeList(request_shape_list)

    newrequest = predict_pb2.PredictRequest()
    newrequest.model_spec.name = request.model_spec.name
    newrequest.model_spec.signature_name = request.model_spec.signature_name
    newrequest.inputs['images'].CopyFrom(tf.contrib.util.make_tensor_proto(tensor_util.MakeNdarray(request.inputs['images'])[:new_request_shape_list[0]], shape=[new_request_shape_list[0]]))
    # newrequest.inputs['images'] = request.inputs['images']

    result = stub.Predict(newrequest, 10.0)
    # print(result)

    return result


    # newrequest = predict_pb2.PredictRequest()
    # newrequest.model_spec.name = 'inception'
    # newrequest.model_spec.signature_name = 'predict_images'
    
    # iteration_list = [1, 10, 100]
    # for iteration in iteration_list:
    #   start = time.time()
    #   for i in range(iteration):
    #     # Send request
    #     with open(request.input_path, 'rb') as f:
    #       # See prediction_service.proto for gRPC request/response details.
    #       data = f.read()
    #       newrequest.inputs['images'].CopyFrom(
    #           tf.contrib.util.make_tensor_proto(data, shape=[1]))
    #       result = stub.Predict(newrequest, 10.0)  # 10 secs timeout
    #       # print(result)

    #   end = time.time()

    #   print("It takes %s sec to run %d images for Inception" % (str(end - start), iteration))

    # return tomtest_pb2.OlympianReply(output_message="Good Job!")




def main(_):
  # host, port = FLAGS.server.split(':')
  # channel = implementations.insecure_channel(host, int(port))
  # stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  # request = predict_pb2.PredictRequest()
  # request.model_spec.name = 'inception'
  # request.model_spec.signature_name = 'predict_images'
  
  # iteration_list = [1]
  # for iteration in iteration_list:
  #   start = time.time()
  #   for i in range(iteration):
  #     # Send request
  #     with open(FLAGS.image, 'rb') as f:
  #       # See prediction_service.proto for gRPC request/response details.
  #       data = f.read()
  #       request.inputs['images'].CopyFrom(
  #           tf.contrib.util.make_tensor_proto(data, shape=[1]))
  #       result = stub.Predict(request, 10.0)  # 10 secs timeout
  #       print(result)

  #   end = time.time()

  #   print("It takes %s sec to run %d images for Inception" % (str(end - start), iteration))
  

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