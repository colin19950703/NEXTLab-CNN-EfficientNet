#! /usr/bin/env python
# coding=utf-8
from easydict import EasyDict as edict


__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C

# NEXTLAB options
__C.NEXTLAB                      = edict()

__C.NEXTLAB.CLASSES_JSON               = "./data/classes/car.json"
__C.NEXTLAB.CLASSES_JSON_AUGMENTATION  = "./data/dataset/car_aug.json"

__C.NEXTLAB.BACKUP_TRAIN_IMAGES_PATH   = "./data/backup/dataset/train/image"
__C.NEXTLAB.BACKUP_TRAIN_LABELS_PATH   = "./data/backup/dataset/train/label"
__C.NEXTLAB.BACKUP_VALID_IMAGES_PATH   = "./data/backup/dataset/valid/image"
__C.NEXTLAB.BACKUP_VALID_LABELS_PATH   = "./data/backup/dataset/valid/label"

__C.NEXTLAB.ALL_DATAFRAME_PATHS        = "./data/classes/json_paths_All.csv"
__C.NEXTLAB.CLASS_DICT_DATAFRAME       = "./data/classes/class_dict.csv"
__C.NEXTLAB.MODEL_PATH                  = "./data/model"

__C.NEXTLAB.SEED                        = 42
__C.NEXTLAB.IMAGE_SIZE                  = [224,224]
__C.NEXTLAB.IMAGE_CHANNELS              = 3
__C.NEXTLAB.BATCH_SIZE                  = 128
__C.NEXTLAB.EPOCH                       = 5
__C.NEXTLAB.NUM_CLASSES                 = 322
