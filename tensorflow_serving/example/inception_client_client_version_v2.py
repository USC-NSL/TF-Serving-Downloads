# Copyright 2015, Google Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following disclaimer
# in the documentation and/or other materials provided with the
# distribution.
#     * Neither the name of Google Inc. nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""The Python implementation of the GRPC helloworld.Greeter client."""

# from __future__ import print_function

import grpc
import time

# from tensorflow_serving.apis import tomtest_pb2
# from tensorflow_serving.apis import tomtest_grpc_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import olympian_master_grpc_pb2

import tensorflow as tf

from tensorflow.python.framework import tensor_util

def run():
  channel = grpc.insecure_channel('localhost:50051')
  stub = olympian_master_grpc_pb2.OlympianMasterStub(channel)

  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'inception'
  request.model_spec.signature_name = 'predict_images'

  input_path = '/home/yitao/Documents/TF-Serving-Downloads/cat.jpg'

  iteration_list = [1, 10, 100]
  for iteration in iteration_list:
    start = time.time()
    for i in range(iteration):
      # Send request
      with open(input_path, 'rb') as f:
        # See prediction_service.proto for gRPC request/response details.
        data = f.read()
        request.inputs['images'].CopyFrom(
            tf.contrib.util.make_tensor_proto(data, shape=[1]))
        result = stub.Predict(request, 10.0)  # 10 secs timeout
        # print(result)

    end = time.time()

    print("It takes %s sec to run %d images for Inception" % (str(end - start), iteration))


  # response = stub.CallTest(tomtest_pb2.OlympianRequest(input_path='/home/yitao/Documents/TF-Serving-Downloads/dog.jpg'))
  # print("Greeter client received: " + response.output_message)

def runBatch():
  channel = grpc.insecure_channel('localhost:50051')
  stub = olympian_master_grpc_pb2.OlympianMasterStub(channel)

  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'inception'
  request.model_spec.signature_name = 'predict_images'

  batchSize = 100
  image = "/home/yitao/Documents/TF-Serving-Downloads/dog.jpg"

  iteration_list = [2]
  for iteration in iteration_list:
    
    for i in range(iteration):
      image_data = []
      start = time.time()
      for j in range(batchSize):
        with open(image, 'rb') as f:
          image_data.append(f.read())

      request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image_data, shape=[len(image_data)]))

      result = stub.Predict(request, 10.0)  # 10 secs timeout
      
      result_classes = tensor_util.MakeNdarray(result.outputs['classes'])
      print("received result of shape: %s" % str(result_classes.shape))
      # print(result_classes)
      
      end = time.time()
      duration = end - start
      print("it takes %s sec" % str(duration))

if __name__ == '__main__':
  # run()
  runBatch()
