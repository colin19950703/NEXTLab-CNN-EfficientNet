####custom module####
import models
from core import utils
from core.config import cfg
####################
import os
import random
import numpy as np
import tensorflow as tf

import pandas as pd
from glob import glob
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
# class configure:
#     seed = 42
#     EPOCH = 25
#     BATCH_SIZE = 32
#     lr = 1e-4
#     decay_rate = lr / EPOCH

#     modelname = 'Vgg16_FCN'
#     model = models.Vgg16_FCN()
#     target_size = modelinfo[modelname]


def seed_fixing(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)

seed_fixing(cfg.NEXTLAB.SEED)

def main():
    #사전에 증강한 이미지 정보가 담긴 json파일을 읽어와 DataFrame으로 변경
    path = cfg.NEXTLAB.CLASSES_JSON_AUGMENTATION
    df = pd.read_json(path, "r", encoding='UTF8')
    df = df.T.rename_axis('class_name').reset_index()

    ## DataFrame을 8:2로 스플릿
    df_train , df_valid = utils.train_test_split_custom(df, train_size=0.8, random_state=cfg.NEXTLAB.SEED)
    df_train.to_csv(cfg.NEXTLAB.CLASS_DATAFRAME_PATHS, mode='w',encoding='euc-kr', index=False)
    NUM_TRAIN = len(df_train)
    NUM_TEST = len(df_valid)

    print(f'number of train : {NUM_TRAIN}\nnumber of validation : {NUM_TEST}')
    
    ##학습을 위한 데이터셋 생성
    args = dict(rescale=1./255,
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest')

    train_generator, validation_generator = utils.dataset_generater(df_train, df_valid, args)

    ##모델 생성
    model = models.EfficientNet(version = 'v2', 
                                model_type='b0', 
                                input_shape= cfg.NEXTLAB.IMAGE_SIZE, 
                                n_channels = cfg.NEXTLAB.IMAGE_CHANNELS,
                                n_classes = cfg.NEXTLAB.NUM_CLASSES)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    model.summary()

    #콜백 함수 정의
    save_model = cfg.NEXTLAB.MODEL_PATH + '/EfficientNet.h5'

    checkpoint = ModelCheckpoint(save_model,
                                 monitor='val_loss',
                                 save_best_only=True,
                                 mode='auto'
                                )

    reduceLR = ReduceLROnPlateau(monitor='val_loss',
                                 factor=0.5,
                                 patience=2
                                 )

    earlystopping = EarlyStopping(monitor='val_loss',
                                  patience=5,
                                 )

    history = model.fit(train_generator, 
                 steps_per_epoch=NUM_TRAIN/cfg.NEXTLAB.BATCH_SIZE,
                 validation_data=validation_generator, 
                 validation_steps=NUM_TEST/cfg.NEXTLAB.BATCH_SIZE,
                 epochs=cfg.NEXTLAB.EPOCH,
                 callbacks = [checkpoint, earlystopping],
                 verbose = 1)

    #ACC 그래프 저장
    utils.save_historygraph(history)

if __name__ == '__main__':
    main()
