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

def run_model_in_parallel(stub, run_num_per_thread, batch_size, model_name, thread_id):
  request = predict_pb2.PredictRequest()
  request.model_spec.name = model_name
  request.model_spec.signature_name = 'predict_images'

  # if (model_name == "inception"):
  #   folder_name = "000"
  # else:
  #   folder_name = "001"
  folder_name = str(thread_id).zfill(3)

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
    result = stub.Predict(request, 60.0 + thread_id * 10)

    end = time.time()

    duration = (end - start)
    print("[%s-%d-%d] it takes %s sec" % (model_name, thread_id, i, str(duration)))
    duration_sum += duration

  # print("For model %s-%d run in parallel, it takes %s sec to run a batch of %d images over %d runs on average" % (model_name, thread_id, str(duration_sum), batch_size, run_num_per_thread))



def main(_):
  host, port = FLAGS.server.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

  model_names = ["inception", "caffe_googlenet", "caffe_resnet50", "caffe_resnet101", "caffe_resnet152", "caffe_alexnet", "caffe_vgg"]
  batch_sizes = [150, 200, 144, 128, 100, 256, 120]
  # model_names = ["inception"]
  # batch_sizes = [150]

  geneate_cost_model_run_num = 13

  for i in range(len(model_names)):
    model_name = model_names[i]
    batch_size = batch_sizes[i]
    generate_cost_model_for_this_model(stub, geneate_cost_model_run_num, batch_size, model_name)

  # time.sleep(1)
  # print("...5")
  # time.sleep(1)
  # print("...4")
  # time.sleep(1)
  # print("...3")
  # time.sleep(1)
  # print("...2")
  # time.sleep(1)
  # print("...1")
  # time.sleep(1)


  # run_num_per_thread = 1
  # client_per_model = 2

  # t_pool = []
  # for i in range(client_per_model):
  #   for j in range(len(model_names)):
  #     model_name = model_names[j]
  #     batch_size = batch_sizes[j]
  #     t_pool.append(threading.Thread(target = run_model_in_parallel, args = (stub, run_num_per_thread, batch_size, model_name, i + client_per_model * j)))

  # start = time.time()

  # for t in t_pool:
  #   t.start()

  # for t in t_pool:
  #   t.join()

  # end = time.time()

  # print('\nFinished!')
  # print("[Sum] the total running time for these %d CNNs is %s" % (len(model_names) * client_per_model, str(end - start)))




  run_num_per_thread = 1
  client_per_model = 2

  for i in range(len(model_names)):
    t_pool = []
    model_name = model_names[i]
    batch_size = batch_sizes[i]
    for j in range(client_per_model):
      t_pool.append(threading.Thread(target = run_model_in_parallel, args = (stub, run_num_per_thread, batch_size, model_name, i + client_per_model * j)))

    start = time.time()

    for t in t_pool:
      t.start()
    for t in t_pool:
      t.join()

    end = time.time()

    print("The total running time to run two concurrent %s of batch size %d is %s" % (model_name, batch_size, str(end - start)))

if __name__ == '__main__':
  tf.app.run()
