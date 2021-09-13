import random
from EfficientNet_codestates.core.config import batch_size, SEED

import pandas as pd
from tqdm.notebook import tqdm
from sklearn.utils import shuffle

from tensorflow.keras.preprocessing.image import ImageDataGenerator

class TrainDataLoader:  # Load Data(approxmately 300MB)
    def __init__(self, train_size=0.8,
                 width=224, height=224, dim=3,
                 origin_path='./data/dataset/',
                 file_path='train/',
                 json_name=str):

        self.train_size = train_size
        self.test_size = 1 - train_size

        self.origin_path = origin_path
        self.file_path = file_path
        self.json_name = json_name

        self.path_list = [origin_path + path + json_name for path in file_path if type(file_path) == list]

        self.width = width
        self.height = height
        self.dim = dim

        self.shape = (self.width, self.height, self.dim)

        self.train_df = pd.DataFrame(
            data={'class_name': []}
        )
        self.valid_df = pd.DataFrame(
            data={'class_name': []}
        )

    def iterator(self, dataframe, index, target):
        lists = dataframe[target].iloc[index]
        random.Random(SEED).shuffle(lists)
        return lists

    def load_data(self):
        train_df = self.train_df
        valid_df = self.valid_df

        df = pd.read_json(self.origin_path + self.json_name,
                          "r", encoding='UTF8').T.rename_axis('class_name').reset_index()

        for index, class_name in tqdm(enumerate(df['class_name'])):
            image_paths = self.iterator(df, index, target='image_path')

            split_threshold = round(len(image_paths) * self.train_size)

            image_paths_train = image_paths[:split_threshold]
            image_paths_test = image_paths[split_threshold:]

            for path in image_paths_train:
                train_df.loc[path] = df['class_name'].iloc[index]
            for path in image_paths_test:
                valid_df.loc[path] = df['class_name'].iloc[index]

        return train_df, valid_df

    def load_multiple_data(self):
        train_df = self.train_df
        valid_df = self.valid_df

        train_df_init = pd.read_json(self.path_list[0], "r", encoding='UTF8').T.rename_axis('class_name').reset_index()
        valid_df_init = pd.read_json(self.path_list[1], "r", encoding='UTF8').T.rename_axis('class_name').reset_index()

        for index, class_name in tqdm(enumerate(train_df_init['class_name'])):
            image_paths_train = self.iterator(train_df_init, index, target='image_path')

            for path in image_paths_train:
                train_df.loc[path] = train_df_init['class_name'].iloc[index]

        for index, class_name in tqdm(enumerate(valid_df_init['class_name'])):
            image_paths_test = self.iterator(valid_df_init, index, target='image_path')

            for path in image_paths_test:
                valid_df.loc[path] = valid_df_init['class_name'].iloc[index]

        return train_df, valid_df

    def shuffle_index(self, train_df, valid_df):
        train_df = train_df.reset_index()
        valid_df = valid_df.reset_index()

        train_df = train_df.rename(columns={"index": "path"})
        valid_df = valid_df.rename(columns={"index": "path"})

        if type(self.file_path) == list:
            path_train = self.origin_path + self.file_path[0]
            path_test = self.origin_path + self.file_path[1]

        else:
            path_train = self.origin_path
            path_test = self.origin_path

        train_df['path'] = train_df['path'].str.replace('./data/dataset/', path_train)
        valid_df['path'] = valid_df['path'].str.replace('./data/dataset/', path_test)

        train_df = shuffle(train_df, random_state=SEED)
        valid_df = shuffle(valid_df, random_state=SEED)

        return train_df, valid_df

    def validate_data(self, train_df, valid_df):
        tester = pd.concat([train_df, valid_df])
        print(f"train size: {len(train_df)}\ntest_size: {len(valid_df)}\n")
        print(
            f"total size: {len(tester)}\ndrop duplicated data: {len(tester.drop_duplicates().reset_index(drop=True))}")

    def load(self):
        if type(self.file_path) == str or None:
            train_df, valid_df = self.load_data()
        else:
            train_df, valid_df = self.load_multiple_data()

        train_df, valid_df = self.shuffle_index(train_df, valid_df)
        self.validate_data(train_df, valid_df)
        return train_df, valid_df


def data_generator(train_df, valid_df, height, width):
    train_generator = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_generator = ImageDataGenerator(rescale=1. / 255)

    train_data = train_generator.flow_from_dataframe(
        train_df,
        x_col='path',
        y_col='class_name',
        target_size=(height, width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_data = test_generator.flow_from_dataframe(
        valid_df,
        x_col='path',
        y_col='class_name',
        target_size=(height, width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_data, test_data


def call_data(train_size=0.8, json_name=str, origin_path=str, file_path=str or list or None, width=224, height=224):
    loader = TrainDataLoader(train_size=train_size, json_name=json_name,
                             origin_path=origin_path, file_path=file_path,
                             width=width, height=height)
    train_df, test_df = loader.load()

    train_data, test_data = data_generator(train_df, test_df, loader.height, loader.width)
    return train_data, test_data


def testdata_generator(path, height, width):
    test_generator = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_data = test_generator.flow_from_directory(
        path,
        target_size=(height, width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    return test_data
