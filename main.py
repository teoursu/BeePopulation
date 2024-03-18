import json
import os
import shutil

import pandas as pd
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator
from scipy.stats._mstats_basic import trim

sns.set_style('darkgrid')
from sklearn.model_selection import train_test_split
import configparser


with open('config.json') as f:
    config = json.load(f)

parent_dir = config['Paths']['parent_dir']
img_dir = config['Paths']['img_dir']
csv_dir = config['Paths']['csv_dir']

df = pd.read_csv(csv_dir)
df = df.drop(['date', 'time', 'location', 'zip code', 'health', 'pollen_carrying', 'caste'], axis=1)
df.columns = ['filepaths', 'labels']
df['filepaths'] = df['filepaths'].apply(lambda x: os.path.join(img_dir, x))

train_df, dummy_df = train_test_split(df, train_size=.8, shuffle=True, random_state=123, stratify=df['labels'])
valid_df, test_df = train_test_split(dummy_df, train_size=.5, shuffle=True, random_state=123,
                                     stratify=dummy_df['labels'])
print('train_df lenght: ', len(train_df), '  test_df length: ', len(test_df), '  valid_df length: ', len(valid_df))
groups = train_df.groupby('labels')
class_count = len(groups)
print(' There are ', class_count, ' classes in the dataframe')
print('{0:^30s} {1:^13s}'.format('CLASS', 'IMAGE COUNT'))
for label in train_df['labels'].unique():
    group = groups.get_group(label)
    print('{0:^30s} {1:^13s}'.format(label, str(len(group))))

max_samples = 400
min_samples = 0
column = 'labels'
train_df = trim(train_df, max_samples, min_samples, column)


def balance(train_df, max_samples, min_samples, column, working_dir, image_size):
    train_df = train_df.copy()
    aug_dir = os.path.join(working_dir, 'aug')
    if os.path.isdir(aug_dir):
        shutil.rmtree(aug_dir)
    os.mkdir(aug_dir)
    for label in train_df['labels'].unique():
        dir_path = os.path.join(aug_dir, label)
        os.mkdir(dir_path)
    total = 0
    gen = ImageDataGenerator(horizontal_flip=True, rotation_range=20, width_shift_range=.2,
                             height_shift_range=.2, zoom_range=.2)
    groups = train_df.groupby('labels')
    for label in train_df['labels'].unique():
        group = groups.get_group(label)
        sample_count = len(group)
        if sample_count < max_samples:
            aug_img_count = 0
            delta = max_samples - sample_count
            target_dir = os.path.join(aug_dir, label)
            aug_gen = gen.flow_from_dataframe(group, x_col='filepaths', y_col=None, target_size=image_size,
                                              class_mode=None, batch_size=1, shuffle=False,
                                              save_to_dir=target_dir, save_prefix='aug-', color_mode='rgb',
                                              save_format='jpg')
            while aug_img_count < delta:
                images = next(aug_gen)
                aug_img_count += len(images)
            total += aug_img_count
    print('Total Augmented images created= ', total)
    if total > 0:
        aug_fpaths = []
        aug_labels = []
        classlist = os.listdir(aug_dir)
        for klass in classlist:
            classpath = os.path.join(aug_dir, klass)
            flist = os.listdir(classpath)
            for f in flist:
                fpath = os.path.join(classpath, f)
                aug_fpaths.append(fpath)
                aug_labels.append(klass)
        Fseries = pd.Series(aug_fpaths, name='filepaths')
        Lseries = pd.Series(aug_labels, name='labels')
        aug_df = pd.concat([Fseries, Lseries], axis=1)
        train_df = pd.concat([train_df, aug_df], axis=0).reset_index(drop=True)

    print(list(train_df['labels'].value_counts()))
    return train_df


working_dir = r'./'
img_size = (70, 70)
train_df = balance(train_df, max_samples, min_samples, column, working_dir, img_size)
