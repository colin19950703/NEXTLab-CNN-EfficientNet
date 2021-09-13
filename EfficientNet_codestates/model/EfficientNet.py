import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from .model import EfficientNetB0, EfficientNetB1, EfficientNetB2
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

class SEBlock(keras.Model):
    def __init__(self, input_size, r=4):
        super(SEBlock, self).__init__(trainable=True)
        self.excitation = keras.Sequential([
            layers.Dense(input_size * r),
            layers.Activation(activation='swish'),
            layers.Dense(input_size),
            layers.Activation(activation='sigmoid')
        ])
        self.squeeze = layers.GlobalAveragePooling2D(keepdims=True, data_format='channels_last')

    def call(self, x):
        x = self.squeeze(x)
        x = self.excitation(x)
        return x

class MBConv(keras.Model):
    __expand = 6

    def __init__(self, input, output, strides,
                 kernel_size, se_scale=4, p=0.5):
        super(MBConv, self).__init__(trainable=True)
        self.p = tf.convert_to_tensor(p, dtype=float) if (input == output) else tf.convert_to_tensor(1, dtype=float)

        self.residual = keras.Sequential([
            layers.Conv2D(filters=input * MBConv.__expand, kernel_size=1,
                          strides=1, padding='same', use_bias=False),
            layers.BatchNormalization(momentum=0.99, epsilon=1e-3),
            layers.Activation(activation='swish'),

            layers.Conv2D(filters=input * MBConv.__expand, kernel_size=kernel_size,
                          strides=1, padding='same', use_bias=False),
            layers.BatchNormalization(momentum=0.99, epsilon=1e-3),
            layers.Activation(activation='swish'),
        ])
        self.se = SEBlock(input * self.__expand, se_scale)
        self.project = keras.Sequential([
            layers.Conv2D(output, kernel_size=1, strides=1, padding='valid', use_bias=False),
            layers.BatchNormalization(momentum=0.99, epsilon=1e-3)
        ])
        self.shortcut = (strides == 1 and (input == output))

    def call(self, x):
        if self.fit:
            if not tfp.distributions.Bernoulli(self.p):
                return x
        x_shortcut = x
        x_residual = self.residual(x)
        x_se = self.se.call(x_residual)
        print(f"residual: {tf.shape(x_residual)}\nse: {tf.shape(x_se)}")

        x = x_se * x_residual
        x = self.project(x)

        if self.shortcut:
            x = x_shortcut + x
        return x


class SepConv(keras.Model):
    __expand = 1
    def __init__(self, input, output, strides,
                 kernel_size, se_scale=4, p=0.5):
        super(SepConv, self).__init__(trainable=True)
        self.p = tf.convert_to_tensor(p, dtype=float) if (input == output) else tf.convert_to_tensor(1, dtype=float)

        self.residual = keras.Sequential([
            layers.Conv2D(filters=input * SepConv.__expand, kernel_size=kernel_size,
                          strides=1, padding='same', use_bias=False),
            layers.BatchNormalization(momentum=0.99, epsilon=1e-3),
            layers.Activation(activation='swish')
            ])
        self.se = SEBlock(input * self.__expand, se_scale)
        self.project = keras.Sequential([
            layers.Conv2D(input * SepConv.__expand, kernel_size=1, strides=1, padding='valid', use_bias=False),
            layers.BatchNormalization(momentum=0.99, epsilon=1e-3)
        ])
        self.shortcut = (strides == 1 and (input == output))

    def call(self, x):
        if self.fit:
            if not tfp.distributions.Bernoulli(self.p):
                return x
        x_shortcut = x
        x_residual = self.residual(x)
        x_se = self.se.call(x_residual)
        print(f"residual: {tf.shape(x_residual)}\nse: {tf.shape(x_se)}")

        x = x_se * x_residual
        x = self.project(x)

        if self.shortcut:
            x = x_shortcut + x
        return x

    class EfficientNet(keras.Model):
        def __init__(self):
            return None


class Model:
    def __init__(self, model_name='model', b=str, add_block_list=None,
                 weights='imagenet', width=224, height=224, dim=3,
                 activation='softmax', none_trainable_layers=0, classes=int,
                 dropout_rate=0.5, unlock_layer_list=['multiply_16'],
                 epochs=60, batch_size=128, smoothing_rate=0, monitor='val_loss'):
        """
        b = EfficientnetB(b), int.
        model = layers you want to add. list, classes.
        none_trainable_layers: select none-trainable layers. int.
        classes = number of classes
        """
        self.b = b
        self.weights = weights

        self.width = width
        self.height = height
        self.dim = dim

        self.shape = (width, height, dim)

        self.epochs = epochs
        self.batch_size = batch_size
        self.monitor = monitor

        self.none_trainable_layers = none_trainable_layers
        self.add_block_list = add_block_list
        self.classes = classes
        self.dropout_rate = dropout_rate
        self.unlock_layer_list = unlock_layer_list

        self.activation = activation

        self.model = {'b0': EfficientNetB0(include_top=False, weights='imagenet', input_shape=self.shape,
                                           classes=self.classes),
                      'b1': EfficientNetB1(include_top=False, weights='imagenet', input_shape=self.shape,
                                           classes=self.classes),
                      'b2': EfficientNetB2(include_top=False, weights='imagenet', input_shape=self.shape,
                                           classes=self.classes)
                      }

        self.optimizer = {'adam': Adam(learning_rate=0.001, name='Adam'),
                          "SGD": tf.keras.optimizers.SGD(learning_rate=0.001,
                                                         momentum=0.0,
                                                         nesterov=False,
                                                         name="SGD")
                          }

        self.loss = {'Top5Acc': tf.keras.metrics.TopKCategoricalAccuracy(k=3),
                     'CategoricalCrossentropy': tf.keras.losses.CategoricalCrossentropy(label_smoothing=smoothing_rate)
                     }

        self.model_name = f'checkpoint-epoch-{epochs}-batch-{batch_size}-trial-001.h5'

        self.checkpoint = ModelCheckpoint(model_name,
                                          monitor=monitor,
                                          verbose=1,
                                          save_best_only=True,
                                          mode='auto'
                                          )

        self.reduceLR = ReduceLROnPlateau(monitor=monitor,
                                          factor=0.5,
                                          patience=2
                                          )

        self.earlystopping = EarlyStopping(monitor=monitor,
                                           patience=5,
                                           )

    def set_untrainable_layers(self):
        model = self.model[self.b]

        if self.none_trainable_layers == 'all':
            model.trainable = False

        else:
            for layer in model.layers[:self.none_trainable_layers]:
                layer.trainable = False

        for layer in model.layers[-20:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True

        for layer in model.layers:
            if layer.name in self.unlock_layer_list:
                set_trainable = True

        return model

    def add_model(self, model, block_list):
        base = model

        if block_list != None:
            assert type(block_list) == list
            for block in block_list:
                base.add(block)

        return base

    def call(self, pooling_layers='average_pooling', pretrained=False, path_weights=None):
        model_pre = self.set_untrainable_layers()

        model = tf.keras.Sequential([])
        model.add(tf.keras.layers.Input(shape=self.shape))
        model.add(self.add_model(model_pre, self.add_block_list))
        if pooling_layers == 'maxpooling':
            model.add(tf.keras.layers.GlobalMaxPooling2D())
        else:
            model.add(tf.keras.layers.GlobalAveragePooling2D())
        model.add(tf.keras.layers.Dropout(self.dropout_rate))
        model.add(tf.keras.layers.Dense(self.classes, activation=self.activation))
        if pretrained == True:
            model.load_weights(path_weights)
        return model

