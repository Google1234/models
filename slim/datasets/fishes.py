# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the flowers dataset.

The dataset scripts used to create the dataset can be found at:
tensorflow/models/slim/datasets/download_and_convert_flowers.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from datasets import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = 'fishes_%s_*.tfrecord'

SPLITS_TO_SIZES = {'train': 44310, 'validation': 350, 'test':1592}

_NUM_CLASSES = 8


def get_split(split_name, dataset_dir, read_boxes,file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading flowers.

  Args:
    split_name: A train/validation split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/validation split.
  """
  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)
  
  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)
  
  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader
  if split_name!="test": # train or validation datasets
    if read_boxes==False:
        keys_to_features = {
          'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
          'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
          'image/class/label': tf.FixedLenFeature(
              [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        }

        items_to_handlers = {
          'image': slim.tfexample_decoder.Image(),
          'label': slim.tfexample_decoder.Tensor('image/class/label'),
        }
        _ITEMS_TO_DESCRIPTIONS = {
            'image': 'A color image of varying size.',
            'label': 'A single integer between 0 and 4,train or validation datasets has this attribute'
        }
    else:
        keys_to_features = {
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
            'image/class/label': tf.FixedLenFeature(
                [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
            'image/name': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/boxes_num': tf.FixedLenFeature([], tf.int64, default_value=0),
            'image/boxes': tf.VarLenFeature(tf.float32),
        }

        items_to_handlers = {
            'image': slim.tfexample_decoder.Image(),
            'label': slim.tfexample_decoder.Tensor('image/class/label'),
            'name': slim.tfexample_decoder.Tensor('image/name'),
            'count': slim.tfexample_decoder.Tensor('image/boxes_num'),
            'boxes': slim.tfexample_decoder.Tensor('image/boxes'),
        }
        _ITEMS_TO_DESCRIPTIONS = {
            'image': 'A color image of varying size.',
            'label': 'A single integer between 0 and 7',
            'name': 'the name of the image ',
            'boxes_count': 'how many box in this image',
            'boxes': 'the Annotated boxes of a image ',
        }

  else :                # test datasets
    keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
      'image/filename': tf.FixedLenFeature((), tf.string, default_value='loss_name'),
    }

    items_to_handlers = {
      'image': slim.tfexample_decoder.Image(),
      'name':  slim.tfexample_decoder.Tensor('image/filename'),
    }
    _ITEMS_TO_DESCRIPTIONS = {
        'image': 'A color image of varying size.',
        'name': 'the name of the image ',
    }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = None
  if dataset_utils.has_labels(dataset_dir):
    labels_to_names = dataset_utils.read_label_file(dataset_dir)

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=_NUM_CLASSES,
      labels_to_names=labels_to_names)
