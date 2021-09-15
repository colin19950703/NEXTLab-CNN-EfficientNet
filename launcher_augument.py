from core import utils
from core.config import cfg

from imgaug import augmenters as iaa
from argparse import RawTextHelpFormatter

import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Please enter augument image number',
                                     formatter_class=RawTextHelpFormatter
                                     )
    parser.add_argument('--aug', default=100, type=int, dest='n_aug',
                        help= 'number of augument'),
    parser.add_argument('--limit', default=True, type=bool, dest='b_limit',
                        help= 'flag to limit maxium number of images'),

    return parser.parse_args()

def main(args):    
    ##Agumentaion 시퀀셜을 설정합니다.
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    seq = iaa.Sequential([iaa.Cutout(nb_iterations=(1, 5), size=0.2, squared=False),
                          sometimes(iaa.Affine(rotate=(-25, 25))),
                          sometimes(iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))),          
                          sometimes(iaa.pillike.Autocontrast((10, 20), per_channel=True)),
                          iaa.GammaContrast(0.5, 1.0),
                          iaa.Fliplr(0.5),
                         ])

    ##"./data/classes/car.json"의 경로에서 car.json (클래스당 카운팅 된 json 파일)을 불러옵니다.
    df_classes = pd.read_json(cfg.NEXTLAB.CLASSES_JSON, "r", encoding='UTF8')
    df_classes = df_classes.T.rename_axis('class_name').reset_index()
    print("num of class", len(df_classes))
    ##Aunmentaion을 진행합니다.
    utils.augmentaion(df_classes, seq, args.n_aug , args.b_limit)
        
#python launcher_augument.py --aug=100
if __name__ == '__main__':
    main(parse_args())