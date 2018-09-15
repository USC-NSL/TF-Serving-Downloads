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

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

import cv2
import numpy as np
import time
import os

from scipy.misc import imread

tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS

# ========================================================================================================================
from easydict import EasyDict as edict

cfg = edict()

cfg.stride = 8.0
cfg.weigh_part_predictions = False
cfg.weigh_negatives = False
cfg.fg_fraction = 0.25
cfg.weigh_only_present_joints = False
cfg.mean_pixel = [123.68, 116.779, 103.939]
cfg.shuffle = True
cfg.snapshot_prefix = "snapshot"
cfg.log_dir = "log"
cfg.global_scale = 1.0
cfg.location_refinement = False
cfg.locref_stdev = 7.2801
cfg.locref_loss_weight = 1.0
cfg.locref_huber_loss = True
cfg.optimizer = "sgd"
cfg.intermediate_supervision = False
cfg.intermediate_supervision_layer = 12
cfg.regularize = False
cfg.weight_decay = 0.0001
cfg.mirror = False
cfg.crop = False
cfg.crop_pad = 0
cfg.scoremap_dir = "test"
cfg.dataset = ""
cfg.dataset_type = "default"  # options: "default", "coco"
cfg.use_gt_segm = False
cfg.batch_size = 1
cfg.video = False
cfg.video_batch = False
cfg.sparse_graph = []
cfg.pairwise_stats_collect = False
cfg.pairwise_stats_fn = "pairwise_stats.mat"
cfg.pairwise_predict = False
cfg.pairwise_huber_loss = True
cfg.pairwise_loss_weight = 1.0
cfg.tensorflow_pairwise_order = True

import yaml
from easydict import EasyDict as edict

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        #if k not in b:
        #    raise KeyError('{} is not a valid config key'.format(k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config from file filename and merge it into the default options.
    """
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, cfg)

    return cfg

def load_config(filename = "pose_cfg.yaml"):
    if 'POSE_PARAM_PATH' in os.environ:
        filename = os.environ['POSE_PARAM_PATH'] + '/' + filename
    return cfg_from_file(filename)
# ========================================================================================================================

def data_to_input(data):
    return np.expand_dims(data, axis=0).astype(float)

def extract_cnn_output(outputs_np, cfg, pairwise_stats = None):
    scmap = outputs_np['part_prob']
    scmap = np.squeeze(scmap)
    locref = None
    pairwise_diff = None
    if cfg.location_refinement:
        locref = np.squeeze(outputs_np['locref'])
        shape = locref.shape
        print("[Yitao] locref.shape = %s" % str(shape))
        # print("[Yitao] ")
        locref = np.reshape(locref, (shape[0], shape[1], -1, 2))
        locref *= cfg.locref_stdev
    if cfg.pairwise_predict:
        pairwise_diff = np.squeeze(outputs_np['pairwise_pred'])
        shape = pairwise_diff.shape
        pairwise_diff = np.reshape(pairwise_diff, (shape[0], shape[1], -1, 2))
        num_joints = cfg.num_joints
        for pair in pairwise_stats:
            pair_id = (num_joints - 1) * pair[0] + pair[1] - int(pair[0] < pair[1])
            pairwise_diff[:, :, pair_id, 0] *= pairwise_stats[pair]["std"][0]
            pairwise_diff[:, :, pair_id, 0] += pairwise_stats[pair]["mean"][0]
            pairwise_diff[:, :, pair_id, 1] *= pairwise_stats[pair]["std"][1]
            pairwise_diff[:, :, pair_id, 1] += pairwise_stats[pair]["mean"][1]
    return scmap, locref, pairwise_diff

def argmax_pose_predict(scmap, offmat, stride):
    """Combine scoremat and offsets to the final pose."""
    num_joints = scmap.shape[2]
    pose = []
    for joint_idx in range(num_joints):
        maxloc = np.unravel_index(np.argmax(scmap[:, :, joint_idx]),
                                  scmap[:, :, joint_idx].shape)
        offset = np.array(offmat[maxloc][joint_idx])[::-1] if offmat is not None else 0
        pos_f8 = (np.array(maxloc).astype('float') * stride + 0.5 * stride +
                  offset)
        pose.append(np.hstack((pos_f8[::-1],
                               [scmap[maxloc][joint_idx]])))
    return np.array(pose)




def main(_):
  cfg = load_config("/home/yitao/Documents/fun-project/tensorflow-related/pose-tensorflow/demo/pose_cfg.yaml")

  host, port = FLAGS.server.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'pose_tensorflow'
  request.model_spec.signature_name = 'predict_images'

  # image_name = "/home/yitao/Documents/fun-project/tf-pose-estimation/images/p1.jpg"
  file_name = "/home/yitao/Documents/fun-project/tensorflow-related/pose-tensorflow/demo/image.png"


  # iteration_list = [1]
  iteration_list = [15, 1, 10]
  for iteration in iteration_list:
    start = time.time()
    for i in range(iteration):

      # print("[%s] start pre-processing" % str(time.time()))

      image = imread(file_name, mode='RGB')
      data = data_to_input(image)

      request.inputs['tensor_inputs'].CopyFrom(
        tf.contrib.util.make_tensor_proto(data, dtype = np.float32, shape=data.shape))

      # print("[%s] start processing" % str(time.time()))

      result = stub.Predict(request, 10.0)

      # print("[%s] finish processing" % str(time.time()))

      # tensor_locref = result.outputs["tensor_locref"]
      # print(tensor_locref.float_val[64484:64512])

      outputs_np = dict()
      outputs_np["locref"] = np.reshape(result.outputs["tensor_locref"].float_val, (-1, 64, 36, 28))
      outputs_np["part_prob"] = np.reshape(result.outputs["tensor_part_prob"].float_val, (-1, 64, 36, 14))

      scmap, locref, _ = extract_cnn_output(outputs_np, cfg)
      pose = argmax_pose_predict(scmap, locref, cfg.stride)

      # print(pose)

    end = time.time()
    print("It takes %s sec to run %d images for tf-openpose" % (str(end - start), iteration))




if __name__ == '__main__':
  tf.app.run()
