from core import utils
from core.config import cfg
from argparse import RawTextHelpFormatter

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Please enter the value for dropping class',
                                     formatter_class=RawTextHelpFormatter
                                     )
    parser.add_argument('--n_limit', default=-1, type=int, dest='n_limit',)
    return parser.parse_args()

def main(args):
    utils.write_class_names(args.n_limit)
    print("Done!!")


#python launcher_classname.py --n_limit 30
if __name__ == '__main__':
    main(parse_args())