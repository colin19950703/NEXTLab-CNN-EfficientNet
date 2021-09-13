from imgaug import augmenters as iaa
from tqdm import tqdm
import pandas as pd
import json
import glob
import os
import PIL
from PIL import Image 
import numpy as np
import math
import argparse

class FilterJson:
    def __init__(self, filter_dict, data_path, label_path):
        self.filter_dict = filter_dict
        self.data_path = data_path
        self.label_path = label_path

    def get_paths(self):
        __SUBDIR_JSON = '/**/**/**.json'
        paths = glob.glob(self.data_path + self.label_path + __SUBDIR_JSON)
        return paths

    def filter_jsons(self, paths=list, filter = True):
        json_dataframe = pd.DataFrame()
        print('Read Jsons!!')
        for i, path  in enumerate(tqdm(paths)) : 
            with open(path, "r", encoding='UTF8') as json_file:
                json_data = json.load(json_file)
                json_data = json_data['car']
                dir = os.path.splitext(json_data['imagePath'])
                data = {}
                data['image_path'] = dir[0]
                data['brand'] = json_data['attributes']['brand']
                data['color'] = json_data['attributes']['color']
                data['model'] = json_data['attributes']['model']
                data['year']  = json_data['attributes']['year']

                if json_data['attributes']['brand'] in self.filter_dict['foreign_brand']:
                    json_dataframe = json_dataframe.append(data, ignore_index=True)
                elif json_data['attributes']['brand'] in self.filter_dict['domestic_brand'] : 
                    year = json_data['attributes']['year'].split('-')
                    if year[1] == '' :  year[1] = year[0]
                    
                    min, max = int(year[0]), int(year[1])
                    
                    if max < 2010 or min >= 2020 : 
                        json_dataframe = json_dataframe.append(data, ignore_index=True)
        return json_dataframe   
    
    def count_brand(self, dataframe) :
        df_list = []

        for brand in self.filter_dict['foreign_brand'] :
            brand_dataframe = pd.DataFrame()
            isbrand = dataframe['brand'] == brand
            brand_dataframe = dataframe[isbrand].reset_index(drop=True)
            if len(brand_dataframe.index) > 0 :
                df_list.append(brand_dataframe)
        
        for brand in self.filter_dict['domestic_brand'] :
            brand_dataframe = pd.DataFrame()
            isbrand = dataframe['brand'] == brand
            brand_dataframe = dataframe[isbrand].reset_index(drop=True)
            if len(brand_dataframe.index) > 0 :
                df_list.append(brand_dataframe)

        return df_list

    def edit_json(self, path, index = 1):
        load_path = self.data_path + self.label_path + '/' + path + '.json'
        save_path = self.data_path + self.label_path + '/' + path + f'_aug{index}.json'
        with open(load_path, "r", encoding='UTF8') as json_file:
            json_data = json.load(json_file)
            json_data['car']['imagePath'] = json_data['car']['imagePath'].replace(".jpg", f'_aug{index}.jpg')
            #ex) 해치백/쉐보레_대우/해치백_마티즈-16.jpg -> 해치백/쉐보레_대우/해치백_마티즈-16_aug10.jpg
        
        with open(save_path, "w", encoding='UTF8') as json_file:
            json.dump(json_data, json_file)

def Augumentaion(args):
    n_aug = args.n_aug
    data_path = args.data_path
    image_path = args.image_path
    label_path = args.label_path
    filter  =   {'year' : [2010,2021], 
                'foreign_brand' : ['BMW', '닛산', '도요타', '랜드로버', '렉서스', '미니', 
                                   '벤츠', '볼보', '아우디', '인피니티', '재규어', '지프', 
                                   '테슬라', '포드', '포르쉐', '폭스바겐', '푸조','혼다'],
                'domestic_brand': ['기아자동차','르노삼성','쉐보레_대우','쌍용자동차','제네시스','현대자동차']}

    seq     =   iaa.Sequential([iaa.Cutout(nb_iterations=2),
                                iaa.Affine(rotate=(-25, 25)),
                                iaa.Fliplr(0.5),
                                iaa.GammaContrast((0.5, 2.0))])

    FI      =   FilterJson(filter, data_path, label_path)
    paths   =   FI.get_paths()          #Converting, filtered json files to data frames
    df      =   FI.filter_jsons(paths)  #Converting, filtered json files to data frames
    df_list =   FI.count_brand(df)      #brand-seperated dataframe
    
    print('Data Augument!!')
    for df in tqdm(df_list, total=len(df_list)) :
        for per, path in  enumerate(df['image_path']) :
            print("[",df['brand'][0],"] progress ::: " , round((per+1)*100/len(df['image_path'].index),2)) ##tqdm ERROR
            images  =   [] #empty list for saving images to augmentation.
            image   = np.array(PIL.Image.open(data_path + image_path+ '/'+ path + '.jpg')) 
            iter    = math.ceil(n_aug / len(df.index)) #iter is number of augmentation.
            for _ in range(iter) :
                images.append(image) #Upload  same number of images as the number of augmentation's iteration .
            
            images_aug = seq(images=images) 
            
            for index, image in enumerate(images_aug) :
                image = Image.fromarray(image)
                image.save(data_path + image_path + '/' + path + f'_aug{index}.jpg')
                FI.edit_json(path, index)   

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, dest='data_path',
                        default=_DEFAULT_DATA_PATH,
                        help='bsolute path of the saved data path'),
    parser.add_argument('--image-path', type=str, dest='image_path',
                        default=_DEFAULT_IMAGE_PATH,
                        help='path of the saved image.'
                             '(example = image)'
                             f'(defaults: {_DEFAULT_IMAGE_PATH})'),
    parser.add_argument('--label-name', type=str, dest='label_path',
                        default=_DEFAULT_LABEL_PATH,
                        help='path of the saved image.'
                            f'(defaults: {_DEFAULT_LABEL_PATH})')
    parser.add_argument('--n-aug', type=int, dest='n_aug', 
                        default=15200,
                        help='number of image to augumentaion TFRecord into.'
                         '(defaults: 15200)')

    return parser.parse_args()

if __name__ == '__main__':
    Augumentaion(parse_args())
    print("Done!!")