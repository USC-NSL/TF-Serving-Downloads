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
import cv2

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

  print("For model %s-%d run in parallel, it takes %s sec to run a batch of %d images over %d runs on average" % (model_name, thread_id, str(duration_sum), batch_size, run_num_per_thread))

def generate_yolo_cost_model(stub, geneate_cost_model_run_num, batch_size, model_name):
  request = predict_pb2.PredictRequest()
  request.model_spec.name = model_name
  request.model_spec.signature_name = 'predict_images'

  duration_sum = 0.0

  for i in range(geneate_cost_model_run_num):
    image_data = []
    start = time.time()
    for j in range(batch_size):
      image_name = "/home/yitao/Downloads/inception-input/%s/dog-%s.jpg" % (str(i % 100).zfill(3), str(j).zfill(3))
      np_img = cv2.imread(image_name)
      resized_img = cv2.resize(np_img, (448, 448))
      np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
      np_img = np_img.astype(np.float32)
      np_img = np_img / 255.0 * 2 - 1
      np_img = np.reshape(np_img, (1, 448, 448, 3))

      image_data.append(np_img)

    batch = np.concatenate(image_data, 0)

    mid = time.time()

    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(batch, shape=[batch_size, 448, 448, 3]))

    result = stub.Predict(request, 60.0)

    end = time.time()

    duration = (end - start)
    print("it takes %s sec with %s sec for image loading" % (str(duration), str(mid - start)))
    if (i != 0 and i != 3 and i != 8):
      duration_sum += duration

  print("For model %s cost model generation, it takes %s sec to run a batch of %d images over %d runs on avereage" % (model_name, str(duration_sum / (geneate_cost_model_run_num - 3)), batch_size, (geneate_cost_model_run_num - 3)))

def run_yolo_in_parallel(stub, run_num_per_thread, batch_size, model_name, thread_id):
  request = predict_pb2.PredictRequest()
  request.model_spec.name = model_name
  request.model_spec.signature_name = 'predict_images'

  folder_name = str(thread_id).zfill(3)

  duration_sum = 0.0

  for i in range(run_num_per_thread):
    image_data = []
    start = time.time()
    for j in range(batch_size):
      image_name = "/home/yitao/Downloads/inception-input/%s/dog-%s.jpg" % (folder_name, str(j).zfill(3))
      np_img = cv2.imread(image_name)
      resized_img = cv2.resize(np_img, (448, 448))
      np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
      np_img = np_img.astype(np.float32)
      np_img = np_img / 255.0 * 2 - 1
      np_img = np.reshape(np_img, (1, 448, 448, 3))

      image_data.append(np_img)

    batch = np.concatenate(image_data, 0)
    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(batch, shape=[batch_size, 448, 448, 3]))

    result = stub.Predict(request, 60.0)

    end = time.time()

    duration = (end - start)
    print("[%s-%d-%d] it takes %s sec" % (model_name, thread_id, i, str(duration)))
    duration_sum += duration

  print("For model %s-%d run in parallel, it takes %s sec to run a batch of %d images over %d runs on average" % (model_name, thread_id, str(duration_sum), batch_size, run_num_per_thread))


def main(_):
  host, port = FLAGS.server.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

  model_name_one = "inception"
  model_name_two = "caffe_googlenet"
  model_name_three = "caffe_resnet152"

  batch_size_model_one = 100
  batch_size_model_two = 100
  batch_size_model_three = 100
  batch_size_model_four = 32

  model_name_four = "yolo_tiny_batch_%s" % str(batch_size_model_four).zfill(3)


  geneate_cost_model_run_num = 13

  # generate_cost_model_for_this_model(stub, geneate_cost_model_run_num, batch_size_model_one, model_name_one)
  # generate_cost_model_for_this_model(stub, geneate_cost_model_run_num, batch_size_model_two, model_name_two)
  # generate_cost_model_for_this_model(stub, geneate_cost_model_run_num, batch_size_model_three, model_name_three)
  generate_yolo_cost_model(stub, geneate_cost_model_run_num, batch_size_model_four, model_name_four)

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


  # # for sr_info(0, 15) and sr_info(1, 15), we force them to overlap from beginning
  # run_num_per_thread = 1
  # client_per_model = 5

  # t_pool = []
  # for i in range(client_per_model):
  #   t_pool.append(threading.Thread(target = run_model_in_parallel, args = (stub, run_num_per_thread, batch_size_model_one, model_name_one, i)))
  #   t_pool.append(threading.Thread(target = run_model_in_parallel, args = (stub, run_num_per_thread, batch_size_model_two, model_name_two, i + client_per_model)))
  #   t_pool.append(threading.Thread(target = run_model_in_parallel, args = (stub, run_num_per_thread, batch_size_model_three, model_name_three, i + client_per_model * 2)))
  #   t_pool.append(threading.Thread(target = run_yolo_in_parallel, args = (stub, run_num_per_thread, batch_size_model_four, model_name_four, i + client_per_model * 3)))

  # start = time.time()
  # # t_pool[0].start()
  # # t_pool[1].start()
  # for t in t_pool:
  #   t.start()

  # # t_pool[0].join()
  # # t_pool[1].join()
  # for t in t_pool:
  #   t.join()

  # end = time.time()

  # print('\nFinished!')
  # print("[Sum] the total running time for %d Inception and %d Caffe-ResNet is %s" % (client_per_model, client_per_model, str(end - start)))

if __name__ == '__main__':
  tf.app.run()
