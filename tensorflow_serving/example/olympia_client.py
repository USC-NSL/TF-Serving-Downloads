from __future__ import print_function

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

import time
import threading
import sys

tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS

import os.path as osp
import numpy as np

def generate_cost_model_for_this_model(stub, geneate_cost_model_run_num, batch_size, model_name):
  request = predict_pb2.PredictRequest()
  request.model_spec.name = model_name
  request.model_spec.signature_name = 'predict_images'

  duration_sum = 0.0

  for i in range(geneate_cost_model_run_num):
    image_data = []

    start = time.time()

    for j in range(batch_size):
      image = "/home/yitao/Downloads/inception-input/000/dog-%s.jpg" % (str(j).zfill(3))
      with open(image, 'rb') as f:
        image_data.append(f.read())
    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image_data, shape=[len(image_data)]))
    result = stub.Predict(request, 60.0)

    end = time.time()

    duration = (end - start)
    print("it takes %s sec" % str(duration))
    if (i != 0 and i != 3 and i != 8):
      duration_sum += duration

  print("For model %s cost model generation, it takes %s sec to run a batch of %d images over %d runs on avereage" % (model_name, str(duration_sum / (geneate_cost_model_run_num - 3)), batch_size, (geneate_cost_model_run_num - 3)))

def run_model_in_parallel(stub, run_num_per_thread, batch_size, model_name):
  request = predict_pb2.PredictRequest()
  request.model_spec.name = model_name
  request.model_spec.signature_name = 'predict_images'

  if (model_name == "inception"):
    folder_name = "000"
  else:
    folder_name = "001"

  duration_sum = 0.0

  for i in range(run_num_per_thread):
    image_data = []

    start = time.time()

    for j in range(batch_size):
      image = "/home/yitao/Downloads/inception-input/%s/dog-%s.jpg" % (folder_name, str(j).zfill(3))
      with open(image, 'rb') as f:
        image_data.append(f.read())
    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image_data, shape=[len(image_data)]))
    result = stub.Predict(request, 60.0)

    end = time.time()

    duration = (end - start)
    print("it takes %s sec" % str(duration))
    duration_sum += duration

  print("For model %s run in parallel, it takes %s sec to run a batch of %d images over %d runs on average" % (model_name, str(duration_sum), batch_size, run_num_per_thread))


def main(_):
  host, port = FLAGS.server.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

  model_name_one = "inception"
  model_name_two = "caffe_googlenet"
  model_name_three = "caffe_resnet152"

  batch_size = 200
  geneate_cost_model_run_num = 13

  generate_cost_model_for_this_model(stub, geneate_cost_model_run_num, batch_size, model_name_one)
  # generate_cost_model_for_this_model(stub, geneate_cost_model_run_num, batch_size, model_name_two)
  generate_cost_model_for_this_model(stub, geneate_cost_model_run_num, batch_size, model_name_three)


  # for sr_info(0, 15) and sr_info(1, 15), we force them to overlap from beginning
  run_num_per_thread = 1

  t_pool = []
  t_pool.append(threading.Thread(target = run_model_in_parallel, args = (stub, run_num_per_thread, batch_size, model_name_one)))
  t_pool.append(threading.Thread(target = run_model_in_parallel, args = (stub, run_num_per_thread, batch_size, model_name_three)))

  start = time.time()
  t_pool[0].start()
  t_pool[1].start()

  t_pool[0].join()
  t_pool[1].join()

  end = time.time()

  print('\nFinished!')
  print("[Sum] the total running time for Inception and Caffe-ResNet is %s" % str(end - start))  

if __name__ == '__main__':
  tf.app.run()
