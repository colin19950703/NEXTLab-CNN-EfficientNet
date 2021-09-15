from core.config import cfg

from tqdm import tqdm
import os
import glob
import numpy as np
import random
import json
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import PIL
from PIL import Image 
from tensorflow.keras.preprocessing import image
import cv2



from skimage.transform import resize

############################################################################################################################################################################
### car.json 생성 관련 함수 작성자 이재웅                                                                                                                                                 ###
############################################################################################################################################################################
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

def write_class_names(limit=-1):

    ##classes폴더에 train's json정보와 valid's json정보가 담긴 dataframe이 있는지 검사하고 없다면, 생성해줍니다.
    try :
        dataframe_All = pd.read_csv(cfg.NEXTLAB.ALL_DATAFRAME_PATHS, encoding='euc-kr')
        
    except :
        paths_train = get_paths(cfg.NEXTLAB.BACKUP_TRAIN_LABELS_PATH, 'json') 
        dataframe_train = get_jsons(cfg.NEXTLAB.BACKUP_TRAIN_IMAGES_PATH, paths_train)
        paths_valid = get_paths(cfg.NEXTLAB.BACKUP_VALID_LABELS_PATH, 'json') 
        dataframe_valid = get_jsons(cfg.NEXTLAB.BACKUP_VALID_IMAGES_PATH, paths_valid)
        #dataframe 합치기
        dataframe_All = pd.concat([dataframe_train, dataframe_valid], axis=0, ignore_index=True)
        dataframe_All.to_csv(cfg.NEXTLAB.ALL_DATAFRAME_PATHS, mode='w',encoding='euc-kr', index=False)

    

    #'brand','model','year'를 조합해 클래스를 생성합니다.
    dataframe_All['class_name'] = dataframe_All['brand']+'/'+dataframe_All['model']+'/'+dataframe_All['year']

    # ##이미지의 수가 너무 적은 클래스를 Drop하기 위해서 클래스당 이미지의 수를 count합니다.
    counts = dataframe_All['class_name'].value_counts().to_frame().reset_index()
    counts.columns = ['class_name', 'count']
    dataframe_All = pd.merge(dataframe_All, counts, on= 'class_name')

       
    # ##dataframe_train과 dataframe_valid를 class_name기준으로 병합합니다.
    df = dataframe_All.groupby(['class_name','brand','model','year','count'])['image_path'].apply(','.join).reset_index()

    #'image_path'를 list형태로 바꿔서 dataframe에 저장합니다. 
    df.image_path = df.image_path.str.split(',') 

    #이미지의 수가 너무 적은 클래스를 Drop합니다.
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
### 기타함수 작성자 이재웅                                                                                                                                                                ###
############################################################################################################################################################################

def generate_newpath(path, class_name, index = -1):
    path_split = path.split('/')

    if 'valid' in path_split :
        if index < 0 :
            path_split[7] = path_split[7].replace('.jpg','(2).jpg')
        else :
            path_split[7] = path_split[7].replace(f'_aug{index}.jpg',f'(2)_aug{index}.jpg')
            
    newpath = path_split[0]+'/'+path_split[1]+'/'+path_split[2]+'/'+class_name     

    if not os.path.exists(newpath):
        os.makedirs(newpath)                

    newpath = newpath + '/'+path_split[7]
    return newpath

def augmentaion(df, seq, n_aug, limit_flag=True):  
    for df_index in range(len(df)):
        class_name = df['class_name'].iloc[df_index].replace('/','_')
        image_paths = df['image_path'].iloc[df_index]
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
            nonagued_img_path = generate_newpath(nonagued_img_path, class_name)
            image.save(nonagued_img_path)
            nonagued_img_paths.append(nonagued_img_path)

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
                agued_img_path = generate_newpath(agued_img_path, class_name, i)              
                image.save(agued_img_path) ##./data/dataset_aug/train/image/**/**/**_aug_{i}.jpg    
                agued_img_paths.append(agued_img_path)
        #증강한 이미지에 대응하는 json파일을 수정 및 저장합니다.
        df['image_path'].iloc[df_index] = nonagued_img_paths + agued_img_paths
        print(f"[{df_index}] class::[{df['class_name'].iloc[df_index]}] is done!")
        
    class_names = df.set_index('class_name').T.to_dict()
    with open(cfg.NEXTLAB.CLASSES_JSON_AUGMENTATION, 'w', encoding='utf-8') as make_file:
        json.dump(class_names, make_file, indent="\t", ensure_ascii=False)   
        
##################################################################################################################
# 데이터 셋 생성 관련 함수 작성자 이재웅                                                                            #
##################################################################################################################
def train_test_split_custom(df, train_size = 0.8, random_state=42):

    df_train = pd.DataFrame(data={'class_name':[]})
    df_test = pd.DataFrame(data={'class_name':[]})

    for index, class_name in enumerate(tqdm(df['class_name'],total=len(df))) : 
        image_paths = df['image_path'].iloc[index]
        random.Random(random_state).shuffle(image_paths)
        image_paths_train = image_paths[: round(len(image_paths) * train_size)]
        image_paths_test = image_paths[round(len(image_paths) * train_size):]

        for path in image_paths_train :
            df_train.loc[path] = df['class_name'].iloc[index]
        for path in image_paths_test :
            df_test.loc[path] = df['class_name'].iloc[index] 

    df_train = df_train.reset_index().rename(columns={"index": "path"})
    df_test = df_test.reset_index().rename(columns={"index": "path"})
    return df_train, df_test


def dataset_generater(df_train, df_valid ,arg = dict()):
    train_datagen = ImageDataGenerator(**arg)
    # Note that the validation data should not be augmented!
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
            df_train,
            x_col = 'path',
            y_col = 'class_name',
            target_size=cfg.NEXTLAB.IMAGE_SIZE,
            batch_size=cfg.NEXTLAB.BATCH_SIZE,
            class_mode='categorical')

    validation_generator = test_datagen.flow_from_dataframe(
            df_valid,
            x_col = 'path',
            y_col = 'class_name',
            target_size=cfg.NEXTLAB.IMAGE_SIZE,
            batch_size=cfg.NEXTLAB.BATCH_SIZE,
            class_mode='categorical')

    return train_generator, validation_generator

##################################################################################################################
# 히스토리 그래프 함수 작성자 이재웅                                                                                #
##################################################################################################################
def save_historygraph(history):
    #Plotting 
    acc = history.history['acc'] 
    val_acc = history.history['val_acc'] 
    loss = history.history['loss'] 
    val_loss = history.history['val_loss'] 

    plt.figure(figsize=(8, 8)) 

    plt.subplot(2, 1, 1) 
    plt.plot(acc, label='Training Accuracy') 
    plt.plot(val_acc, label='Validation Accuracy') 
    plt.legend(loc='lower right') 
    plt.ylabel('Accuracy') 
    plt.ylim([min(plt.ylim()),1]) 
    plt.title('Training and Validation Accuracy') 

    plt.subplot(2, 1, 2) 
    plt.plot(loss, label='Training Loss') 
    plt.plot(val_loss, label='Validation Loss') 
    plt.legend(loc='upper right') 
    plt.ylabel('Cross Entropy') 
    plt.ylim([0,10]) 
    plt.title('Training and Validation Loss') 
    plt.xlabel('epoch') 

    save_path = cfg.NEXTLAB.MODEL_PATH + '/history.png'
    plt.savefig(save_path)

##################################################################################################################
# 히트맵 생성 작성자 이재웅                                                                                        #
##################################################################################################################

def make_gradcam_heatmap(img_array, model, pre_trained,last_conv_layer_name, classifier_layer_names):
    
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer  = model.get_layer(pre_trained).get_layer(last_conv_layer_name)
    conv_model       = keras.Model(model.get_layer(pre_trained).inputs, last_conv_layer.output)
    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)

    classifier_model = keras.Model(classifier_input, x)
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer  
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = conv_model(img_array)
        tape.watch(last_conv_layer_output)
        
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    
    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]

    # is our saliency heatmap of class activation
    saliency = np.mean(last_conv_layer_output, axis=-1)
    saliency = np.maximum(saliency, 0) / np.max(saliency)

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    pooled_grads = pooled_grads.numpy()
    
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our grad_cam heatmap of class activation
    grad_cam = np.mean(last_conv_layer_output, axis=-1)
    grad_cam = np.maximum(grad_cam, 0) / np.max(grad_cam)

    return grad_cam, saliency

def merge_with_heatmap(original_img, heatmap):
    original_img = np.array(original_img)
    resized_heatmap=resize(heatmap, cfg.NEXTLAB.IMAGE_SIZE)
    resized_heatmap = np.uint8(255*resized_heatmap)
    resized_heatmap = cv2.applyColorMap(resized_heatmap, cv2.COLORMAP_JET)
    resized_heatmap = cv2.cvtColor(resized_heatmap, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(resized_heatmap, 0.7, original_img, 0.5, 6)

def convert_to_heatmap(heatmap):
    heatmap = np.uint8(255*heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

def show_hotmap (img, heatmap, title='Heatmap', alpha=0.6, cmap='jet', axisOnOff='off'):
    '''
    #type(img) =Image
    #type(heatmap) =2d narray
    '''
    resized_heatmap=resize(heatmap, img.size)
    
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.imshow(resized_heatmap, alpha=alpha, cmap=cmap)
    plt.axis(axisOnOff)
    plt.title(title)
    plt.show()

##################################################################################################################
# 테스트 구동 관련 함수 작성자 이재웅                                                                               #
##################################################################################################################
def prepare_single_input(img_path):
    img = image.load_img(img_path, target_size=cfg.NEXTLAB.IMAGE_SIZE)
    img = image.img_to_array(img)
    img /= 255.
    img = np.expand_dims(img, axis= 0) # (1, 224, 224, 3)
    return img

def predict_image(Mymodel, img_path, top_k_num = 3, classes_dict = {}):
    image = prepare_single_input(img_path)
    result = Mymodel.predict([image])[0]

    result = list(result)
    
    classname_list = []
    pred_value_list = []
    for _ in range(top_k_num) :
      index= result.index(max(result))
      classname_list.append(classes_dict[index])
      pred_value_list.append(max(result))
      result[index] = 0.

    return classname_list, pred_value_list