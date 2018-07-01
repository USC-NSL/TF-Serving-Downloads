
# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

import numpy as np
import cv2

import time

tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS

def myFuncWarmUp(stub, i):
  batchSize = 1
  durationSum = 0.0
  runNum = 13

  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'yolo_tiny_batch_%s' % str(batchSize).zfill(3)
  request.model_spec.signature_name = 'predict_images'

  for k in range(runNum):
    image_data = []
    start = time.time()
    for j in range(batchSize):
      image_name = "/home/yitao/Downloads/inception-input/%s/dog-%s.jpg" % (str(i % 100).zfill(3), str(j).zfill(3))
      np_img = cv2.imread(image_name)
      resized_img = cv2.resize(np_img, (448, 448))
      np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
      np_img = np_img.astype(np.float32)
      np_img = np_img / 255.0 * 2 - 1
      np_img = np.reshape(np_img, (1, 448, 448, 3))

      image_data.append(np_img)

    batch = np.concatenate(image_data, 0)
    print(batch.shape)

    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(batch, shape=[batchSize, 448, 448, 3]))

    tmp_result = stub.Predict(request, 10.0)  # 5 seconds
    # print(len(tmp_result.outputs["scores"].float_val))
    end = time.time()
    duration = (end - start)
    print("it takes %s sec" % str(duration))
    if (k != 0 and k != 3 and k != 8):
      durationSum += duration

  print("[Warm up] on average, it takes %s sec to run a batch of %d images over %d runs" % (str(durationSum / (runNum - 3)), batchSize, (runNum - 3)))


  iteration_list = [1, 10, 20, 40]
  for iteration in iteration_list:
    start = time.time()
    for i in range(iteration):
      image_data = []
      for j in range(batchSize):
        image_name = "/home/yitao/Downloads/inception-input/%s/dog-%s.jpg" % (str(i % 100).zfill(3), str(j).zfill(3))
        np_img = cv2.imread(image_name)
        resized_img = cv2.resize(np_img, (448, 448))
        np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        np_img = np_img.astype(np.float32)
        np_img = np_img / 255.0 * 2 - 1
        np_img = np.reshape(np_img, (1, 448, 448, 3))
        image_data.append(np_img)

      batch = np.concatenate(image_data, 0)

      request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(batch, shape=[batchSize, 448, 448, 3]))

      tmp_result = stub.Predict(request, 10.0)  # 5 seconds
    end = time.time()
    print("[iteration = %d] It takes %s sec to run %d images of batch size %d for YOLO-tiny" % (iteration, str(end - start), iteration, batchSize))


def main(_):
  host, port = FLAGS.server.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

  myFuncWarmUp(stub, 0)


if __name__ == '__main__':
  tf.app.run()