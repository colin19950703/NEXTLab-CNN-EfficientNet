import argparse
from functools import partial
import platform

import matplotlib.pyplot as plt
import tensorflow as tf

_DEFAULT_TFRECORD_DIR = ''
# : Example DIR = 'data/tfrecord/001-744.tfrecord'

class TFRecordLoader:
    def __init__(self, tfrecord_path):
        """[summary]
        Args:
            tfrecord_path ([str of path or list of paths]): []
        """
        self.tfrecord_path = tfrecord_path
        self.feature = {
            'image/encoded': tf.io.FixedLenFeature((), tf.string),
            'image/car_class': tf.io.FixedLenFeature((), tf.string),
        }

    def get_tfrecord(self):
        return tf.data.TFRecordDataset(self.tfrecord_path, compression_type='GZIP')

    def _parse_image_function(self, example_proto):
        return tf.io.parse_single_example(example_proto, self.feature)

    def load(self):
        tfrecord = self.get_tfrecord()
        return tfrecord.map(self._parse_image_function)


class TFRecordViewer(TFRecordLoader):
    def __init__(self, n_image=int):
        self.n_image = n_image

        if platform.system() == 'Darwin': #Mac
            plt.rc('font', family='AppleGothic') 
        elif platform.system() == 'Windows': #Window
            plt.rc('font', family='Malgun Gothic') 
        elif platform.system() == 'Linux': #Linux or Colab
            plt.rc('font', family='NanumBarunGothic')
        plt.rcParams['axes.unicode_minus'] = False #resolve minus symbol breaks when using Hangul font

    def _tensor_decode(self, features):
        car_class  = features['image/car_class'].numpy().decode()
        image = features['image/encoded']
        image = tf.io.decode_jpeg(image)
        image = tf.keras.preprocessing.image.array_to_img(image)
        return car_class, image

    def show(self, parsed_tfrecord):
        self.parsed_tfrecord = parsed_tfrecord

        for features in self.parsed_tfrecord.take(self.n_image):          
            car_class, image = \
            self._tensor_decode(features)          
            plt.text(250, 20, 'car_class : '+ car_class)

            plt.imshow(image)
            plt.show()

    def resize_img(self, tensor):
        self.tensor = tensor
        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecord-path', type=str, dest='tfrecord_path',
                        default=_DEFAULT_TFRECORD_DIR,
                        help='relative path of the tfrecord path in the project file.'
                            f'(defaults: {_DEFAULT_TFRECORD_DIR})')
    parser.add_argument('--n-image', type=int, dest='n_image', default=1,
                    help='number of image to show TFRecord into.'
                         '(defaults: 1)')

    return parser.parse_args()

def main(args):
    loader = TFRecordLoader(args.tfrecord_path)
    viewer = TFRecordViewer(args.n_image)
                            
    parsed_tfrecord = loader.load()
    viewer.show(parsed_tfrecord)
    

if __name__ == '__main__':
    main(parse_args())
