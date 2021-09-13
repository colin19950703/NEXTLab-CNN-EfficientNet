#! /usr/bin/env python
# coding=utf-8
from easydict import EasyDict as edict


__C = edict()
# Consumers can get config by: from config import cfg

cfg = __C

# NEXTLAB options
__C.NEXTLAB = edict()

__C.NEXTLAB.CLASSES_JSON = "./data/classes/car.json"

__C.NEXTLAB.BACKUP_TRAIN_IMAGES_PATH = "./data/backup/dataset/train/image"
__C.NEXTLAB.BACKUP_VALID_IMAGES_PATH = "./data/backup/dataset/valid/image"

__C.NEXTLAB.BACKUP_TRAIN_LABELS_PATH = "./data/backup/dataset/train/label"
__C.NEXTLAB.BACKUP_VALID_LABELS_PATH = "./data/backup/dataset/valid/label"

__C.NEXTLAB.TRAIN_TFRECORDS_PATH = "./data/dataset/train/tfrecord"
__C.NEXTLAB.VALID_TFRECORDS_PATH = "./data/dataset/valid/tfrecord"

__C.NEXTLAB.TRAIN_DATAFRAME_PATHS = "./data/classes/json_paths_train.csv"
__C.NEXTLAB.VALID_DATAFRAME_PATHS = "./data/classes/json_paths_valid.csv"


__C.NEXTLAB.N_SHARDS = 1
__C.NEXTLAB.SEED = 42
__C.NEXTLAB.IMAGE_SIZE = [224,224]
__C.NEXTLAB.BATCH_SIZE = 128


_DEFAULT_DATA_PATH = 'data'
_DEFAULT_LABEL_PATH = 'label'
_DEFAULT_OUTPUT_DIR = ''

_DEFAULT_N_SHARDS = 1

batch_size = 32
dropout_rate = 0.2

SEED = 42