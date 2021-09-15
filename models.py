import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.vgg16 import VGG16
import keras_efficientnet_v2 as efficientnet_v2

vgg_weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

def Unet(input_size=(256, 256, 3)) :
    inputs = tf.keras.layers.Input(input_size)
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    #Contraction path
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
    
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
    
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
    
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    #Expansive path 
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    
    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    
    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    
    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    return tf.keras.models.Model(inputs=[inputs], outputs=[outputs])

def Vgg16_Unet(input_size=(224,224,3)):
    inputs = tf.keras.layers.Input(input_size)
    # inputs = Input(shape=input_size, dtype='float32', name='input')

    """Encoder"""
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(0.01))(inputs)
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(0.01))(p1)
    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',)(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
    
    c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(0.01))(p2)
    c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(0.01))(c3)
    c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
    
    c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(0.01))(p3)
    c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(0.01))(c4)
    c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
    
    c5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(0.01))(p4)
    c5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(0.01))(c5)
    c5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    p5 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c5)
    
    """Decoder"""
    u6 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(512, (3, 3), kernel_initializer='he_normal', padding='same')(c6)
    c6 = tf.keras.layers.BatchNormalization(axis=3)(c6)
    c6 = tf.keras.layers.Activation('relu')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(c7)
    c7 = tf.keras.layers.BatchNormalization(axis=3)(c7)
    c7 = tf.keras.layers.Activation('relu')(c7)
    
    u8 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(c8)
    c8 = tf.keras.layers.BatchNormalization(axis=3)(c8)
    c8 = tf.keras.layers.Activation('relu')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(c9)
    c9 = tf.keras.layers.BatchNormalization(axis=3)(c9)
    c9 = tf.keras.layers.Activation('relu')(c9)
    
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    return tf.keras.models.Model(inputs=[inputs], outputs=[outputs])

def Vgg16_FCN(input_size=(224, 224, 3), NUM_OF_CLASSESS = 1) :
    inputs = tf.keras.layers.Input(input_size)
    s = inputs

    #Vgg16 model architecture
    c1 = tf.keras.layers.Conv2DTranspose(64,(3,3), activation='relu', padding='same', name='conv1_1')(s)
    c1 = tf.keras.layers.Conv2DTranspose(64,(3,3), activation='relu', padding='same', name='conv1_2')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2), name='pool1')(c1)

    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu',  padding='same', name='conv2_1')(p1)
    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu',  padding='same', name='conv2_2')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2), name='pool2')(c2)

    c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu',  padding='same', name='conv3_1')(p2)
    c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu',  padding='same', name='conv3_2')(c3)
    c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu',  padding='same', name='conv3_3')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2), name='pool3')(c3)

    c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(p3)
    c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(c4)
    c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(c4)
    p4 = tf.keras.layers.MaxPooling2D((2, 2), name='pool4')(c4)

    c5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(p4)
    c5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(c5)
    c5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(c5)
    p5 = tf.keras.layers.MaxPooling2D((2, 2), name='pool5')(c5)

    #transfer learning - Vgg16
    base_model = tf.keras.Model(s, p5)
    base_model.load_weights(vgg_weights_path, by_name=True, skip_mismatch=True)
    
    ##FCN

    #Upsampling
    
    #This code is upsampling for FCN-32 only.
    c6 = tf.keras.layers.Conv2D( 4096 , (7, 7) , activation='relu' , padding='same', name='conv6_1')(p5)
    c7 = tf.keras.layers.Conv2D( 4096 , (1, 1) , activation='relu' , padding='same', name='1x1_conv')(c6)
    #FCN-32 ouput
    fcn32_o = tf.keras.layers.Conv2DTranspose(NUM_OF_CLASSESS, (32, 32), strides=(32, 32), use_bias=False, name='1x1_conv_32x')(c7)
    fcn32_o = tf.keras.layers.Activation('sigmoid', name='fcn32_o')(fcn32_o)
    
    #The code following this is the normal part of upsampling.
    o = tf.keras.layers.Conv2DTranspose(NUM_OF_CLASSESS, (4,4), strides=(2,2), use_bias=False, name='pool5_up_2x')(p5) # (16, 16, n)
    o = tf.keras.layers.Cropping2D(cropping=(1,1))(o) # (14, 14, n)

    o2 = tf.keras.layers.Conv2D(NUM_OF_CLASSESS, (1,1), activation='relu', padding='same')(p4) # (14, 14, n)
    o = tf.keras.layers.Add()([o, o2]) # (14, 14, n)
    
    #FCN-16 output
    fcn16_o = tf.keras.layers.Conv2DTranspose(NUM_OF_CLASSESS, (16,16), strides=(16,16), use_bias=False)(o)
    fcn16_o = tf.keras.layers.Activation('sigmoid', name='fcn16_o')(fcn16_o)


    o = tf.keras.layers.Conv2DTranspose(NUM_OF_CLASSESS, (4,4), strides=(2,2), use_bias=False, name='fcn16_up_2x')(o) # (30, 30, n)
    o = tf.keras.layers.Cropping2D(cropping=(1,1))(o) # (28, 28, n)

    o2 = tf.keras.layers.Conv2D(NUM_OF_CLASSESS, (1,1), activation='relu', padding='same')(p3) # (28, 28, n)
    o = tf.keras.layers.Add()([o, o2]) # (28, 28, n)

    # upsample up to the size of the original image
    o = tf.keras.layers.Conv2DTranspose(NUM_OF_CLASSESS, (8,8), strides=(8,8), use_bias=False)(o) # (224, 224, n)
    fcn8_o = tf.keras.layers.Activation('sigmoid', name='fcn8_o')(o)
    outputs = fcn8_o

    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    return model

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = tf.keras.layers.Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if activation == True:
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    return x

def residual_block(blockInput, num_filters=16):
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(blockInput)
    x = tf.keras.layers.BatchNormalization()(x)
    blockInput = tf.keras.layers.BatchNormalization()(blockInput)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = tf.keras.layers.Add()([x, blockInput])
    return x

def ResUnet(input_shape=(None, None, 3), dropout_rate=0.1, start_neurons = 16):
    #c -> conv
    #p -> pool
    #cm -> convm
    #d -> deconv
    #u -> upconv
    backbone = tf.keras.applications.ResNet101V2(weights='imagenet', include_top=False, input_shape=input_shape)
    input_layer = backbone.input
    
    c4 = backbone.layers[122].output
    c4 = tf.keras.layers.LeakyReLU(alpha=0.1)(c4)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)
    p4 = tf.keras.layers.Dropout(dropout_rate)(p4)
    
    cm = tf.keras.layers.Conv2D(start_neurons * 32, (3, 3), activation=None, padding="same")(p4)
    cm = residual_block(cm,start_neurons * 32)
    cm = residual_block(cm,start_neurons * 32)
    cm = tf.keras.layers.LeakyReLU(alpha=0.1)(cm)
    
    d4 = tf.keras.layers.Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(cm)
    u4 = tf.keras.layers.concatenate([d4, c4])
    u4 = tf.keras.layers.Dropout(dropout_rate)(u4)
    
    u4 = tf.keras.layers.Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(u4)
    u4 = residual_block(u4,start_neurons * 16)
    u4 = residual_block(u4,start_neurons * 16)
    u4 = tf.keras.layers.LeakyReLU(alpha=0.1)(u4)
    
    d3 = tf.keras.layers.Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(u4)
    c3 = backbone.layers[76].output
    u3 = tf.keras.layers.concatenate([d3, c3])    
    u3 = tf.keras.layers.Dropout(dropout_rate)(u3)
    
    u3 = tf.keras.layers.Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(u3)
    u3 = residual_block(u3,start_neurons * 8)
    u3 = residual_block(u3,start_neurons * 8)
    u3 = tf.keras.layers.LeakyReLU(alpha=0.1)(u3)

    d2 = tf.keras.layers.Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(u3)
    c2 = backbone.layers[30].output
    u2 = tf.keras.layers.concatenate([d2, c2])
        
    u2 = tf.keras.layers.Dropout(0.1)(u2)
    u2 = tf.keras.layers.Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(u2)
    u2 = residual_block(u2,start_neurons * 4)
    u2 = residual_block(u2,start_neurons * 4)
    u2 = tf.keras.layers.LeakyReLU(alpha=0.1)(u2)
    
    d1 = tf.keras.layers.Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(u2)
    c1 = backbone.layers[2].output
    u1 = tf.keras.layers.concatenate([d1, c1])
    
    u1 = tf.keras.layers.Dropout(0.1)(u1)
    u1 = tf.keras.layers.Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(u1)
    u1 = residual_block(u1,start_neurons * 2)
    u1 = residual_block(u1,start_neurons * 2)
    u1 = tf.keras.layers.LeakyReLU(alpha=0.1)(u1)
    
    u0 = tf.keras.layers.Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(u1)   
    u0 = tf.keras.layers.Dropout(0.1)(u0)
    u0 = tf.keras.layers.Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(u0)
    u0 = residual_block(u0,start_neurons * 1)
    u0 = residual_block(u0,start_neurons * 1)
    u0 = tf.keras.layers.LeakyReLU(alpha=0.1)(u0)
    
    u0 = tf.keras.layers.Dropout(dropout_rate/2)(u0)
    output_layer = tf.keras.layers.Conv2D(1, (1,1), padding="same", activation="sigmoid")(u0)    
    
    model = tf.keras.models.Model(input_layer, output_layer)

    return model

def EfficientNet(version = 'v1', model_type = 'b0', input_shape = [224,224], n_channels = 3, n_classes = 322):
    input_shape = (input_shape[0], input_shape[1], n_channels)
    
    if version == 'v1':
        if model_type == 'b0':
          base_model = efficientnet_v2.EfficientNetV1(model_type, input_shape=input_shape, num_classes=0, include_preprocessing=False,  pretrained='imagenet')
        elif model_type == 'b1':
          base_model = efficientnet_v2.EfficientNetV1(model_type, input_shape=input_shape, num_classes=0,include_preprocessing=False, pretrained='imagenet')
  
    elif version == 'v2':
      if model_type == 'b0':
        base_model = efficientnet_v2.EfficientNetV2(model_type, input_shape=input_shape, num_classes=0, include_preprocessing=False, pretrained='imagenet')
      elif model_type == 'b1':
        base_model = efficientnet_v2.EfficientNetV2(model_type, input_shape=input_shape, num_classes=0, include_preprocessing=False, pretrained='imagenet')

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))
    model.add(base_model)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))

    return model