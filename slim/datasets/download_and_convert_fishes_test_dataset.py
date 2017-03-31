from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf
from  datasets.dataset_utils import int64_feature
from  datasets.dataset_utils import bytes_feature 

# Seed for repeatability.
_RANDOM_SEED = 0

tf.app.flags.DEFINE_string(
    'dataset_dir', '/newdisk/first/lidenghui/jt/fishes/test_stg1', 'The directory where the dataset files are stored.')
FLAGS = tf.app.flags.FLAGS

class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

def my_image_to_tfexample(image_data, image_format, height, width, filename):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/filename': bytes_feature(filename),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
  }))
def _get_filenames(dataset_dir):
  """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.

  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """
  fishes_root = os.path.join(dataset_dir, 'photos')
  #fishes_root=dataset_dir
  photo_filenames = []  
  for filename in os.listdir(fishes_root):
    path = os.path.join(fishes_root, filename)
    photo_filenames.append(path)
  print ("image nums:",len(photo_filenames))
  #print (photo_filenames)
  return photo_filenames


def _get_dataset_filename(dataset_dir, split_name):
  output_filename = 'fishes_%s.tfrecord' % (
      split_name)
  return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, dataset_dir):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
  assert split_name in ['test']

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:
        output_filename = _get_dataset_filename(
            dataset_dir, split_name)
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          for i in range(len(filenames)):
            sys.stdout.write('\r>> Converting image %d/%d %s' % (
                i+1, len(filenames),os.path.basename(filenames[i])))
            sys.stdout.flush()
            # Read the filename:
            image_data = tf.gfile.FastGFile(filenames[i], 'r').read()
            try:
              height, width = image_reader.read_image_dims(sess, image_data)
            except:
              print ("Error read image:",filenames[i])
              raise Exception
            example = my_image_to_tfexample(
                image_data, 'jpg', height, width, os.path.basename(filenames[i]))
            tfrecord_writer.write(example.SerializeToString())
  sys.stdout.write('\n')
  sys.stdout.flush()

def _dataset_exists(dataset_dir):
  for split_name in ['test']:
    #for shard_id in range(_NUM_SHARDS):
      output_filename = _get_dataset_filename(
          dataset_dir, split_name)
      if not tf.gfile.Exists(output_filename):
        return False
  return True


def run(dataset_dir):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)
  if _dataset_exists(dataset_dir):
    print('Dataset files already exist. Exiting without re-creating them.')
    return
  #dataset_utils.download_and_uncompress_tarball(_DATA_URL, dataset_dir)
  photo_filenames= _get_filenames(dataset_dir)

  test_filenames = photo_filenames[:]

  _convert_dataset('test', test_filenames,
                   dataset_dir)
  #_clean_up_temporary_files(dataset_dir)
  print('\nFinished converting the Fishes test dataset!')

def main(_):
  run(FLAGS.dataset_dir)


if __name__ == '__main__':
  tf.app.run()

