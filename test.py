from core import utils
from core.config import cfg

import pandas as pd
import numpy as np
from tensorflow import keras
from PIL import Image 
import matplotlib.pyplot as plt
import cv2

from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

def main():
    model_path      = './data/model/EfficientNet.h5'
    test_path       = './data/test/'
    true_classname  = '포르쉐/SUV_카이엔/2015-2018' ###BMW_세단_5시리즈_2014-2016 ->'BMW/세단_5시리즈/2014-2016'
    test_image_path = test_path + true_classname.replace('/','_') + '.jpg'
    

    path = cfg.NEXTLAB.CLASS_DICT_DATAFRAME
    df = pd.read_csv(path, encoding='cp949')

    ## 클래스 불러오기 위한 데이터셋 생성
    args = dict(rescale=1./255)
    classes, _ = utils.dataset_generater(df, df, args)
    classes_dict = classes.class_indices
    classes_dict = dict(map(reversed, classes_dict.items()))
    ##모델 로드
    
    model = keras.models.load_model(model_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    model.summary()

    ##히트맵을 그리기 위한 layer 추출 과정입니다.

    #defalut classifier_layer_names , 기본 classifier_layer은 아래와 같습니다.
    #classifier_layer_names =  ['global_average_pooling2d', 'dropout', 'dense']

    classifier_layers =  model.layers[-3:]
    classifier_layer_names = []
    for layer in classifier_layers:
        classifier_layer_names.append(layer.name)
    last_conv_layer_name   = 'post_swish'
    pre_train= 'EfficientNetV2'

    img = Image.open(test_image_path).resize(size=cfg.NEXTLAB.IMAGE_SIZE)  
    img_array = utils.prepare_single_input(test_image_path)
    #test
    grad_cam, saliency = utils.make_gradcam_heatmap(img_array, model, pre_train,last_conv_layer_name, classifier_layer_names)
    pred_classnames, pred_values = utils.predict_image(model, test_image_path, classes_dict = classes_dict)
    grad_cam_merge = utils.merge_with_heatmap(img, grad_cam)
    saliency_merge = utils.merge_with_heatmap(img, saliency)
    grad_cam = utils.convert_to_heatmap(grad_cam)
    saliency = utils.convert_to_heatmap(saliency)

    #Save grad_cam image
    fig = plt.figure()
    rows = 2
    cols = 3
    ax1 = fig.add_subplot(rows, cols, 1)
    ax1.imshow(np.array(img))
    ax1.set_title('Original Image')
    ax1.axis("off")
    
    ax2 = fig.add_subplot(rows, cols, 2)
    ax2.imshow(grad_cam)
    ax2.set_title('Grad_cam')
    ax2.axis("off")
    
    ax3 = fig.add_subplot(rows, cols, 3)
    ax3.imshow(grad_cam_merge)
    ax3.set_title('Grad_cam_merge')
    ax3.axis("off")
    
    ax3 = fig.add_subplot(rows, cols, 4)
    ax3.text(0.,1.0,f"[실제 클래스]")
    ax3.text(0.,0.9,f"{true_classname}", fontsize=8)
    ax3.text(0.,0.8,f"[예측 클래스 Top 3]")
    ax3.text(0.,0.6,f"{pred_classnames[0]},\n예측률 : {round(float(pred_values[0]),3)}", fontsize=8)
    ax3.text(0.,0.4,f"{pred_classnames[1]},\n예측률 : {round(float(pred_values[1]),3)}", fontsize=8)
    ax3.text(0.,0.2,f"{pred_classnames[2]},\n예측률 : {round(float(pred_values[2]),3)}", fontsize=8)
    ax3.axis("off")

    ax5 = fig.add_subplot(rows, cols, 5)
    ax5.imshow(saliency)
    ax5.set_title('Saliency')
    ax5.axis("off")
    
    ax6 = fig.add_subplot(rows, cols, 6)
    ax6.imshow(saliency_merge)
    ax6.set_title('Saliency_merge')
    ax6.axis("off")

    path = test_image_path[:-4] + '_result.jpg'
    plt.savefig(path, bbox_inces='tight', pad_inches=0, dpi=100)

if __name__ == '__main__':
    main()


