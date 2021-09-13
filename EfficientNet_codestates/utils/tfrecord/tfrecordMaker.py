import os
import glob
import random
from tqdm.notebook import tqdm
from sklearn.utils import shuffle
import argparse
import numpy as np
import math
import pandas as pd
import tensorflow as tf


from EfficientNet_codestates.core.config import _DEFAULT_DATA_PATH, _DEFAULT_LABEL_PATH, _DEFAULT_OUTPUT_DIR, SEED

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def make_example(img_str, car_class):
    """
    :param img_str: byte-encoded image, str
    :param model: car's model, str
    :return: tf.train.Example file
    """
    feature = {'image/encoded': _bytes_feature(img_str),
               'image/car_class': _bytes_feature(car_class)
               }
    return tf.train.Example(features=tf.train.Features(feature=feature))

class TFRecordConverter:
    def __init__(self, origin_path, data_path, label_path, output_dir, n_shards,
                 train_size=.8, valid_size=.2):
        """
        :param origin_path: base path that contains data/label. TYPE:string
        :param data_path: sub-path that contains train data. DEFAULT: data(str)
        :param label_path: sub-path that contains train label. DEFAULT: label(str)
        """
        self.origin_path = origin_path
        self.data_path = data_path
        self.label_path = label_path
        self.output_dir = output_dir
        self.n_shards = n_shards
        self.train_size = train_size
        self.valid_size = valid_size

        self.train_df = pd.DataFrame(
            data={'class_name': []}
            )
        self.valid_df = pd.DataFrame(
            data={'class_name': []}
            )

    __SUBDIR_JSON = '/**/**/**.json'

    def get_paths(self):
        paths = glob.glob(self.origin_path + self.label_path + self.__SUBDIR_JSON)
        return paths

    def iterator(self, dataframe, index, target):
        lists = dataframe[target].iloc[index]
        random.Random(SEED).shuffle(lists)
        return lists


    def get_jsons(self):
        dataframe = pd.read_json(self.label_path, "r", encoding='UTF8')
        dataframe = dataframe.T.rename_axis('class_name').reset_index()

        train_df = self.train_df
        valid_df = self.valid_df

        for index, class_name in tqdm(enumerate(dataframe['class_name'])):
            image_paths = self.iterator(dataframe, index, target='image_path')

            split_threshold = round(len(image_paths) * self.train_size)

            image_paths_train = image_paths[:split_threshold]
            image_paths_test = image_paths[split_threshold:]

            for path in image_paths_train:
                train_df.loc[path] = dataframe['class_name'].iloc[index]
            for path in image_paths_test:
                valid_df.loc[path] = dataframe['class_name'].iloc[index]

        return shuffle(train_df), valid_df

    def shuffle_index(self, train_df, valid_df):
        train_df = train_df.reset_index()
        valid_df = valid_df.reset_index()

        train_df = train_df.rename(columns={"index": "path"})
        valid_df = valid_df.rename(columns={"index": "path"})

        train_df['path'] = train_df['path'].str.replace('./data/dataset/', self.origin_path)
        valid_df['path'] = valid_df['path'].str.replace('./data/dataset/', self.origin_path)

        train_df = shuffle(train_df, random_state=SEED)
        valid_df = shuffle(valid_df, random_state=SEED)

        return train_df, valid_df

    def get_img(self, image_path=str):
        self.image_path = image_path
        files = tf.io.read_file(image_path)
        return tf.image.decode_image(files, channels=3)

    def _get_shard_path(self, shard_id, shard_size, train_or_valid):
        assert train_or_valid in ['train', 'valid']
        return os.path.join(self.origin_path,
                            self.output_dir,
                            f'{train_or_valid}-{shard_id:03d}-{shard_size}.tfrecord')

    def _write_tfrecord_file(self, df, indices, shard_path):
        self.df = df
        self.shard_path = shard_path
        self.indices = indices
        options = tf.io.TFRecordOptions(compression_type='GZIP')
        with tf.io.TFRecordWriter(shard_path, options=options) as out:
            for index in indices:
                car_class = df.class_name.iloc[index].encode()
                decoded_image = open(df.path.iloc[index], 'rb').read()
                example = make_example(decoded_image, car_class)
                out.write(example.SerializeToString())

    def convert(self):
        paths = self.get_paths()
        train_df, valid_df = self.get_jsons()
        train_df, valid_df = self.shuffle_index(train_df, valid_df)

        train_size = len(train_df)
        offset = 0
        shard_size = math.ceil(train_size/self.n_shards)
        cumulative_size = offset + train_size
        for shard_id in range(1, self.n_shards + 1):
            step_size = min(shard_size, cumulative_size - offset)
            shard_path = self._get_shard_path(shard_id, step_size, 'train')
            file_indices = np.arange(offset, offset + step_size)
            self._write_tfrecord_file(train_df, file_indices, shard_path)
            offset += step_size

        valid_size = len(valid_df)
        offset = 0
        shard_size = math.ceil(valid_size/self.n_shards)
        cumulative_size = offset + valid_size
        for shard_id in range(1, self.n_shards + 1):
            step_size = min(shard_size, cumulative_size - offset)
            shard_path = self._get_shard_path(shard_id, step_size, 'valid')
            file_indices = np.arange(offset, offset + step_size)
            self._write_tfrecord_file(valid_df, file_indices, shard_path)
            offset += step_size

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin-path', type=str, dest='origin_path',
                        help='Absolute path of the target path.'
                             '(example = /User/administrator/project-path/contents/)')
    parser.add_argument('--data-path', type=str, dest='data_path',
                        default=_DEFAULT_DATA_PATH,
                        help='relative path of the data path in the project file.'
                            f'(default: {_DEFAULT_DATA_PATH})')
    parser.add_argument('--label-path', type=str, dest='label_path',
                        default=_DEFAULT_LABEL_PATH,
                        help='relative path of the label path in the project file.'
                            f'(defaults: {_DEFAULT_LABEL_PATH})')
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                        default=_DEFAULT_OUTPUT_DIR,
                        help='relative directory in the project'
                             'that tfrecord file will be saved.'
                            f'(defaults:{_DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--n-shards', type=int, dest='n_shards', default=1,
                        help='number of shards to divide dataset TFRecord into.'
                             '(defaults: 1)')
    return parser.parse_args()

def main(args):
    converter = TFRecordConverter(args.origin_path,
                                  args.data_path,
                                  args.label_path,
                                  args.output_dir,
                                  args.n_shards)
    converter.convert()

if __name__ == '__main__':
    main(parse_args())