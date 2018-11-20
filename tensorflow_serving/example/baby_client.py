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

import time
import threading
import sys

import tensorflow as tf

from tensorflow.python.framework import tensor_util

def runBatchWarmUp(stub, batchSize, runNum):
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'inception'
  request.model_spec.signature_name = 'predict_images'

  image = "/home/yitao/Documents/TF-Serving-Downloads/dog.jpg"
  durationSum = 0.0

  for k in range(runNum):
    image_data = []
    start = time.time()
    for j in range(batchSize):
      with open(image, 'rb') as f:
        image_data.append(f.read())

    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image_data, shape=[len(image_data)]))

    try:
      result = stub.Predict(request, 30.0)  # 10 secs timeout  
    except Exception as e:
      print("[Warmup] Failed with %s" % str(e))
    # else:
    #   result_classes = tensor_util.MakeNdarray(result.outputs['classes'])
    #   print("[Warmup] received result of shape: %s" % str(result_classes.shape))
    end = time.time()
    duration = (end - start)
    print("[Warmup] it takes %s sec" % str(duration))
    if (k != 0 and k != 3 and k != 8):
      durationSum += duration

  print("[Warmup] on average, it takes %s sec to run a batch of %d images over %d runs" % (str(durationSum / (runNum - 3)), batchSize, (runNum - 3)))

def runBatchParallel(stub, batchSize, runNum, threadId):
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'inception'
  request.model_spec.signature_name = 'predict_images'

  image = "/home/yitao/Downloads/inception-input/%s/dog-000.jpg" % str(threadId).zfill(3)
  durationSum = 0.0

  for k in range(runNum):
    image_data = []
    start = time.time()
    for j in range(batchSize):
      with open(image, 'rb') as f:
        image_data.append(f.read())

    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image_data, shape=[len(image_data)]))

    try:
      result = stub.Predict(request, 30.0)  # 10 secs timeout  
    except Exception as e:
      print("[Parallel] Failed with %s" % str(e))
    # else:
    #   result_classes = tensor_util.MakeNdarray(result.outputs['classes'])
    #   print("[Parallel] received result of shape: %s" % str(result_classes.shape))
    end = time.time()
    duration = (end - start)
    print("[Parallel] thread-%d's %dth job takes %s sec" % (threadId, k, str(duration)))
    durationSum += duration

  print("[Parallel] thread-%d spent %s sec to run a batch of %d images over %d runs" % (threadId, str(durationSum), batchSize, runNum))

if __name__ == '__main__':
  channel = grpc.insecure_channel('localhost:50051')
  stub = olympian_master_grpc_pb2.OlympianMasterStub(channel)

  batchSize = 100
  warmupRunNum = 13
  parallelClientNum = 1
  parallelRunNum = 10

  runBatchWarmUp(stub, batchSize, warmupRunNum)

  tPool = []
  for i in range(parallelClientNum):
    tPool.append(threading.Thread(target = runBatchParallel, args = (stub, batchSize, parallelRunNum, i)))

  start = time.time()
  for i in range(parallelClientNum):
    t = tPool[i]
    t.start()

  for i in range(parallelClientNum):
    t = tPool[i]
    t.join()

  end = time.time()

  print("\nAll Finished!")
  print("[Parallel] The total running time to run %d concurrent clients, each with %d jobs with batch size of %d is %s" % (parallelClientNum, parallelRunNum, batchSize, str(end - start)))