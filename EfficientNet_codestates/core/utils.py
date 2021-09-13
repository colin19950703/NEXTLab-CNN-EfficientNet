from EfficientNet_codestates.core.config import cfg
from tqdm import tqdm

import os
import glob
import numpy as np
import json
import pandas as pd
import tensorflow as tf
import random
import PIL
from PIL import Image 

############################################################################################################################################################################
### car.json 생성 관련 함수                                                                                                                                               ###
############################################################################################################################################################################
def write_class_names(limit=-1):

    ##classes폴더에 train's json정보가 담긴 dataframe, valid's json정보가 담긴 dataframe이 있는지 검사하고 없다면, 생성해줍니다.
    try :
        dataframe_train = pd.read_csv(cfg.NEXTLAB.TRAIN_DATAFRAME_PATHS, encoding='euc-kr')
        dataframe_valid = pd.read_csv(cfg.NEXTLAB.VALID_DATAFRAME_PATHS, encoding='euc-kr')
    except :
        paths = get_paths(cfg.NEXTLAB.BACKUP_TRAIN_LABELS_PATH, 'json') 
        dataframe_train = get_jsons(cfg.NEXTLAB.BACKUP_TRAIN_IMAGES_PATH, paths)
        dataframe_train.to_csv(cfg.NEXTLAB.TRAIN_DATAFRAME_PATHS, mode='w',encoding='euc-kr', index=False)

        paths = get_paths(cfg.NEXTLAB.BACKUP_VALID_LABELS_PATH, 'json') 
        dataframe_valid = get_jsons(cfg.NEXTLAB.BACKUP_VALID_IMAGES_PATH, paths)
        dataframe_valid.to_csv(cfg.NEXTLAB.VALID_DATAFRAME_PATHS, mode='w',encoding='euc-kr', index=False)

    ##'brand','model','year'를 조합해 클래스를 생성합니다.
    dataframe_train['class_name'] = dataframe_train['brand']+'/'+dataframe_train['model']+'/'+dataframe_train['year']
    
    ##이미지의 수가 너무 적은 클래스를 Drop하기 위해서 클래스당 이미지의 수를 count합니다.
    counts = dataframe_train['class_name'].value_counts().to_frame().reset_index()
    counts.columns = ['class_name', 'count']
    dataframe_train = pd.merge(dataframe_train, counts, on= 'class_name')

    ##train dataframe과 valid dataframe의 병합을 위해 'class_name'과 'image_path_valid'만 남깁니다.
    dataframe_valid['class_name'] = dataframe_valid['brand']+'/'+dataframe_valid['model']+'/'+dataframe_valid['year']
    dataframe_valid.rename(columns={'image_path':'image_path_valid'}, inplace=True)
    dataframe_valid = dataframe_valid[['class_name','image_path_valid']]
    
    ##dataframe_train과 dataframe_valid를 class_name기준으로 병합합니다.
    dataframe_train = dataframe_train.groupby(['class_name','brand','model','year','count'])['image_path'].apply(','.join).reset_index()
    dataframe_valid = dataframe_valid.groupby(['class_name'])['image_path_valid'].apply(','.join).reset_index()
    df = pd.merge(dataframe_train, dataframe_valid, how = "left", on= 'class_name')

    ##'image_path'를 list형태로 바꿔서 dataframe에 저장합니다. valid에 의해서 생긴 결측치는 ""로 대체합니다.
    df.image_path = df.image_path.str.split(',')  
    df.image_path_valid = df.image_path_valid.str.split(',') 
    df = df.fillna("")

    ##이미지의 수가 너무 적은 클래스를 Drop합니다.
    if limit >= 0 :
        drop_index = df[df['count'] < limit].index
        df = df.drop(drop_index).reset_index(drop=True)
    df = df.drop("count", axis=1)
    df = df.reset_index().rename(columns={"index": "label"})

    ##JSON형태로 저장합니다.
    class_names = df.set_index('class_name').T.to_dict()
    with open(cfg.NEXTLAB.CLASSES_JSON, 'w', encoding='utf-8') as make_file:
        json.dump(class_names, make_file, indent="\t", ensure_ascii=False) 

############################################################################################################################################################################
### TFRECORD 생성 관련 함수                                                                                                                                               ###
############################################################################################################################################################################

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def get_paths_tfrecord_shard(tfrecord_path, shard_id, shard_size):
    return os.path.join(tfrecord_path, f'{shard_id:03d}-{shard_size}.tfrecord')

def make_example(path, img_str, brand_name, model, year, class_name, label):
    """
    :param img_str: byte-encoded image, str
    :param brand_name: brand name of car, str
    :param model: car's model, str
    :param year: car's year, str
    :param class_name: car's class name, str
    :param label: car's class's label, int
    :return: tf.train.Example file
    """
    feature = {'image/path': _bytes_feature(path),
               'image/encoded': _bytes_feature(img_str),
               'image/class/brand': _bytes_feature(brand_name),
               'image/class/model': _bytes_feature(model),
               'image/class/year': _bytes_feature(year),
               'image/class/class_name': _bytes_feature(class_name),
               'image/class/label': _int64_feature(label)
               }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def write_tfrecord_file(df, indices, shard_path, TRAIN = True):
    ##tfrecord로 변환합니다.
    options = tf.io.TFRecordOptions(compression_type='GZIP')
    count = 0
    with tf.io.TFRecordWriter(shard_path, options=options) as out:
        
        for index in tqdm(indices):
            ##train 이미지를 변환합니다 image_path 경로를 선택합니다
            if TRAIN == True :
                paths = df.image_path.iloc[index]
            ##valid 이미지를 변환을 위해 image_path_valid 경로를 선택합니다.
            else :
                paths = df.image_path_valid.iloc[index]

            for path_index in range(len(paths)) :
                class_name = df.class_name.iloc[index].encode()
                label = df.label.iloc[index]
                brand = df.brand.iloc[index].encode()
                model = df.model.iloc[index].encode()
                year = df.year.iloc[index].encode()
                file_path = paths[path_index].replace('/backup/','/').encode()
                decoded_image = open(paths[path_index], 'rb').read()       
                example = make_example(file_path, decoded_image, brand, model, year, class_name, label)                        
                out.write(example.SerializeToString())
                count+=1
    old_name = shard_path
    new_name = shard_path.replace(str(len(indices)), str(count)) 
    os.rename(old_name,new_name)   


############################################################################################################################################################################
### 복호화                                                                                                                                                               ###
############################################################################################################################################################################
def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, cfg.NEXTLAB.IMAGE_SIZE)
    return image

def deserialize_example(serialized_string): 
    features = {
        'image/encoded': tf.io.FixedLenFeature((), tf.string),
        'image/class/label': tf.io.FixedLenFeature((), tf.int64),
    }
    example = tf.io.parse_single_example(serialized_string, features)

    label = tf.cast(example['image/class/label'], tf.int64)    
    image = decode_image(example['image/encoded'])
    return image, label

############################################################################################################################################################################
### 기타함수                                                                                                                                                              ###
############################################################################################################################################################################
def get_length(file_name):
        tmp = file_name.split('-')
        tmp = tmp[1].split('.')
        return int(tmp[0])
        
def get_paths(path=str, target=str):
    if target == 'json' :
        __DIR = '/**/**/**.json'
    elif target == 'tfrecord' :
        __DIR = '/*.tfrecord'
    ##assert
    paths = path + __DIR
    paths = glob.glob(path + __DIR)
    return paths

def get_jsons(image_path=str, jsons_paths=list):
    json_dataframe = pd.DataFrame()
    for path in tqdm(jsons_paths):
        with open(path, "r", encoding='UTF8') as json_file:
            json_data = json.load(json_file)
            json_data = json_data['car']
            data = json_data['attributes']
            data['image_path'] = image_path + '/' + json_data['imagePath']
            json_dataframe = json_dataframe.append(data, ignore_index=True)
    return json_dataframe

def get_img(image_path=str):
    files = tf.io.read_file(image_path)
    return tf.image.decode_image(files, channels=3)

def get_tfrecord(path=str):
    tfrecord = tf.data.TFRecordDataset(path, compression_type = 'GZIP')
    return tfrecord.map(parse_image_function)

def parse_image_function(example_proto):
    feature = {
        'image/path': tf.io.FixedLenFeature((), tf.string),
        'image/encoded': tf.io.FixedLenFeature((), tf.string),
        'image/class/brand': tf.io.FixedLenFeature((), tf.string),
        'image/class/model': tf.io.FixedLenFeature((), tf.string),
        'image/class/year': tf.io.FixedLenFeature((), tf.string),
        'image/class/class_name': tf.io.FixedLenFeature((), tf.string),
        'image/class/label': tf.io.FixedLenFeature((), tf.int64),
    }
    return tf.io.parse_single_example(example_proto, feature)

def tensor_decode(features):
    path  = features['image/path'].numpy().decode()
    brand = features['image/class/brand'].numpy().decode()
    model = features['image/class/model'].numpy().decode()
    year  = features['image/class/year'].numpy().decode()
    class_name  = features['image/class/class_name'].numpy().decode()
    label  = tf.cast(features['image/class/label'], tf.int64)
    image = features['image/encoded']
    image = tf.io.decode_jpeg(image)
    image = tf.keras.preprocessing.image.array_to_img(image)
    return path, brand, model, year, image, class_name, label

def augmentaion(df, seq, n_aug, limit_flag=True):  
    for df_index in range(len(df)):
        image_paths = df['image_path'].iloc[df_index]
        image_paths_valid = df['image_path_valid'].iloc[df_index]
        n_image = len(image_paths)
        image_paths_toAgu = image_paths

        
        if (n_image == n_aug) :
            image_paths_toAgu = image_paths
        ## 클래스당 이미지 보유 수가 증강할 이미지의 수보다 작은 경우  
        elif (n_image < n_aug) :
            ## 증강할 이미지 수만큼 원본 이미지 경로를 랜덤하게 뽑아 추가해줍니다.
            while (len(image_paths_toAgu) < n_aug) :
                image_paths_toAgu+=[random.choice(image_paths)]     
        ## 클래스당 이미지 보유 수가 증강할 이미지의 수보다 클 경우 
        elif (n_image > n_aug): 
            ##이미지 최대 수를 n_aug의 수만큼 제한할거면.
            if limit_flag == True :
                ##이미지수를 증강할 이미지수로 변경   
                n_image=n_aug  
            ## 증강할 이미지 수만큼 원본 이미지 경로를 랜덤하게 뽑아 줄여줍니다.
                while (len(image_paths_toAgu) > n_aug) : 
                    image_paths_toAgu.pop(random.randint(0,len(image_paths_toAgu)-1))
                    image_paths_toAgu = set(image_paths_toAgu)
                    image_paths_toAgu = list(image_paths_toAgu)
        
        ##원본 이미지를 dataset에서 열어서 dataset폴더에 저장합니다. (증강 안함)
        agued_img_paths=[]
        nonagued_img_paths = []
        for path in image_paths_toAgu[:n_image] :
            image = np.array(PIL.Image.open(path))
            image = Image.fromarray(image)
            nonagued_img_path = path.replace('/backup/','/')
            image.save(nonagued_img_path)
            nonagued_img_paths.append(nonagued_img_path)

        for path in image_paths_valid :
            image = np.array(PIL.Image.open(path))
            image = Image.fromarray(image)
            image.save(path.replace('/backup/','/'))

        ##가지고 있는 이미지 수가 증강할 이미지 수를 넘지 않는 경우에만 이미지를 증강합니다.
        if n_image !=  n_aug : 
            image_list=[]
            for path in tqdm(image_paths_toAgu[n_image:]) :
                image =  np.array(PIL.Image.open(path))
                image_list.append(image)

            #이미지를 증강합니다.
            images_aug = seq(images=image_list)

            #증강한 이미지를 저장합니다 경로명은 dataset에서 dataset_aug으로 변경하고 확장자 앞에 _aug+숫자 를 붙여줍니다.
            for i, image in enumerate(images_aug) :
                image = Image.fromarray(image)
                agued_img_path = image_paths_toAgu[i].replace('/backup/','/').replace('.jpg',f'_aug{i}.jpg')              
                image.save(agued_img_path) ##./data/dataset_aug/train/image/**/**/**_aug_{i}.jpg    
                agued_img_paths.append(agued_img_path)
        #증강한 이미지에 대응하는 json파일을 수정 및 저장합니다.
        df['image_path'].iloc[df_index] = nonagued_img_paths + agued_img_paths
        print(f"[{df_index}] class::[{df['class_name'].iloc[df_index]}] is done!")
        
    class_names = df.set_index('class_name').T.to_dict()
    with open(cfg.NEXTLAB.CLASSES_JSON, 'w', encoding='utf-8') as make_file:
        json.dump(class_names, make_file, indent="\t", ensure_ascii=False)   

