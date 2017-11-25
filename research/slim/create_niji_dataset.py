import json
import logging
import os
import random
import sys
from typing import Dict, List

import PIL.Image
import tensorflow as tf

_RANDOM_SEED = 0

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('num_train_shards', 100, '')
tf.app.flags.DEFINE_integer('num_validation_shards', 10, '')
tf.app.flags.DEFINE_string('output_dir', None, '')
tf.app.flags.mark_flag_as_required('output_dir')


def _make_bytes_feature(data: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[data]))


def _make_int64_feature(number: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[number]))


def _make_example(image_data: bytes, width: int, height: int, class_id: int) -> tf.train.Example:
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/encoded': _make_bytes_feature(image_data),
                'image/format': _make_bytes_feature(b'jpg'),
                'image/class/label': _make_int64_feature(class_id),
                'image/width': _make_int64_feature(width),
                'image/height': _make_int64_feature(height),
            }
        )
    )


def _build_shard(writer: tf.python_io.TFRecordWriter, index: Dict[str, int]) -> int:
    num_records = 0
    for index_file, (filename, class_id) in enumerate(index.items()):
        sys.stderr.write('%d/%d...\r' % (index_file + 1, len(index)))
        with open(filename, 'rb') as f:
            image_data = f.read()
        try:
            image = PIL.Image.open(filename)
        except Exception:
            logging.exception('Failed to open: %s', filename)
            continue
        if image.mode != 'RGB':
            logging.warning('Skipping %s mode image: %s', image.mode, filename)
            continue
        width, height = image.size
        example = _make_example(image_data, width, height, class_id)
        writer.write(example.SerializeToString())
        num_records += 1
    sys.stderr.write('                        \r')
    return num_records


def _build_split(split_name: str, index: Dict[str, int], num_shards: int) -> int:
    image_filenames = list(index.keys())
    random.shuffle(image_filenames)
    sharded_indices = [
        {image_filenames[i]: index[image_filenames[i]]
         for i in range(index_shard, len(image_filenames), num_shards)}
        for index_shard in range(num_shards)]
    assert sum(len(subindex) for subindex in sharded_indices) == len(index)
    num_records = 0
    for index_shard, subindex in enumerate(sharded_indices):
        output_filename = 'niji_%s_%05d-of-%05d.tfrecord' % (
            split_name, index_shard, num_shards)
        logging.info('Building: %s', output_filename)
        output_path = os.path.join(FLAGS.output_dir, output_filename)
        with tf.python_io.TFRecordWriter(output_path) as writer:
            num_records += _build_shard(writer, subindex)
    return num_records


def _build_dataset(
        index: Dict[str, int],
        num_train_shards: int,
        num_validation_shards: int):
    image_filenames = list(index.keys())
    random.shuffle(image_filenames)
    num_train = len(image_filenames) * 9 // 10
    train_index = {s: index[s] for s in image_filenames[:num_train]}
    validation_index = {s: index[s] for s in image_filenames[num_train:]}
    num_train_records = _build_split('train', train_index, num_train_shards)
    num_validation_records = _build_split(
        'validation', validation_index, num_validation_shards)

    metadata = {
        'counts': {
            'train': num_train_records,
            'validation': num_validation_records,
        },
    }
    with open(os.path.join(FLAGS.output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)

    with open(os.path.join(FLAGS.output_dir, 'labels.txt'), 'w') as f:
        print('0:3D', file=f)
        print('1:2D', file=f)


def _read_index(index_path: str) -> Dict[str, int]:
    index = {}
    with open(index_path) as f:
        for line in f.read().splitlines():
            image_relpath, class_id_str = line.split()
            image_path = os.path.abspath(
                os.path.join(os.path.dirname(index_path), image_relpath))
            class_id = int(class_id_str)
            if image_path.endswith('.jpg'):
                index[image_path] = class_id
    return index


def main(argv: List[str]):
    logging.basicConfig(level=logging.INFO)
    random.seed(_RANDOM_SEED)

    os.makedirs(FLAGS.output_dir, exist_ok=True)

    index = {}
    for index_path in argv[1:]:
        index.update(_read_index(index_path))

    _build_dataset(
        index,
        num_train_shards=FLAGS.num_train_shards,
        num_validation_shards=FLAGS.num_validation_shards)


if __name__ == '__main__':
    tf.app.run()
