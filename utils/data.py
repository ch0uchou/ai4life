# Copyright 2021 Simone Angarano. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
import numpy as np
from mpose import MPOSE
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


labels = { # 20 Classes
    "standing":0,
    "check-watch":1,
    "cross-arms":2,
    "scratch-head":3,
    "sit-down":4,
    "get-up":5,
    "turn":6, 
    "walk":7, 
    "wave1":8,
    "box":9,
    "kick":10,
    "point":11, 
    "pick-up":12,
    "bend":13,
    "hands-clap":14,
    "wave2":15,
    "jog":16,
    "jump":17,
    "pjump":18,
    "run":19}

def load_y(y_path):
    file = open(y_path, 'r')
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    ).reshape(-1)
    file.close()

    # for 0-based indexing
    return y_

def load_X(X_path, n_steps=32):
    file = open(X_path, 'r')
    X_ = np.array(
        [elem for elem in [
            row.split(',') for row in file
        ]],
        dtype=np.float64
    )
    file.close()
    blocks = int(len(X_) / n_steps)

    X_ = np.array(np.split(X_,blocks)).reshape(blocks, n_steps, 17, 2)

    return X_


def load_mpose(dataset, split, verbose=False, legacy=False, logger = None):
    
    if legacy:
        return load_dataset_legacy(data_folder=f'datasets/openpose_bm/split{split}/base_vars/')
    
    # d = MPOSE(pose_extractor=dataset, 
    #                 split=split, 
    #                 preprocess=None, 
    #                 velocities=True, 
    #                 remove_zip=False)
    # logger.save_log(f"dataset info: {d.get_info()} ")
    # X_train, y_train, X_test, y_test = d.get_data()
    # logger.save_log(f"X_train shape: {X_train.shape}")
    # logger.save_log(f"y_train shape: {y_train.shape}")
    # logger.save_log(f"X_train[] shape: {X_train[1000,15,:,:]}")
    if 'legacy' not in dataset:
        # d.reduce_keypoints()
        # X_train, y_train, X_test, y_test = d.get_data()
        # logger.save_log(f"X_train[reduce_keypoints] shape: {X_train.shape}")
        # d.scale_and_center()
        # X_train, y_train, X_test, y_test = d.get_data()
        # logger.save_log(f"X_train[scale_and_center] shape: {X_train.shape}")
        # d.remove_confidence()
        # X_train, y_train, X_test, y_test = d.get_data()
        # logger.save_log(f"X_train[remove_confidence] shape: {X_train.shape}")
        # d.flatten_features()
        # X_train, y_train, X_test, y_test = d.get_data()
        # logger.save_log(f"X_train[flatten_features] shape: {X_train.shape}")
        # #d.reduce_labels()
        # return d.get_data()
        X_train = load_X(f'X.txt')
        y_train = load_y(f'Y.txt')
        X_test = load_X(f'X_test.txt')
        y_test = load_y(f'Y_test.txt')
        for X in [X_train, X_test]:
            seq_list = []
            for seq in X:
                pose_list = []
                for pose in seq:
                    zero_point = (pose[1, :2] + pose[2,:2]) / 2
                    module_keypoint = (pose[7, :2] + pose[8,:2]) / 2
                    scale_mag = np.linalg.norm(zero_point - module_keypoint)
                    if scale_mag < 1:
                        scale_mag = 1
                    pose[:,:2] = (pose[:,:2] - zero_point) / scale_mag
                    pose_list.append(pose)
                seq = np.stack(pose_list)
                seq_list.append(seq)
            X = np.stack(seq_list)
        
        X_train = np.delete(X_train, [], 2)
        X_test = np.delete(X_test, [], 2)
        X_train = X_train.reshape(X_train.shape[0], 32, -1)
        X_test = X_test.reshape(X_test.shape[0], 32, -1)
        return X_train, y_train, X_test, y_test
    
    if 'openpose' in dataset:
        X_train, y_train, X_test, y_test = d.get_data()
        return X_train, transform_labels(y_train), X_test, transform_labels(y_test)
    else:
        return d.get_data()

def random_flip(x, y):
    time_steps = x.shape[0]
    n_features = x.shape[1]
    if not n_features % 2:
        x = tf.reshape(x, (time_steps, n_features//2, 2))

        choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
        if choice >= 0.5:
            x = tf.math.multiply(x, [-1.0,1.0])
    else:
        x = tf.reshape(x, (time_steps, n_features//3, 3))

        choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
        if choice >= 0.5:
            x = tf.math.multiply(x, [-1.0,1.0,1.0])
    x = tf.reshape(x, (time_steps,-1))
    return x, y


def random_noise(x, y):
    time_steps = tf.shape(x)[0]
    n_features = tf.shape(x)[1]
    noise = tf.random.normal((time_steps, n_features), mean=0.0, stddev=0.03, dtype=tf.float64)
    x = x + noise
    return x, y


def one_hot(x, y, n_classes):
    return x, tf.one_hot(y, n_classes)


def kinetics_generator(X, y, batch_size):
    while True:
        ind_list = [i for i in range(X.shape[0])]
        shuffle(ind_list)
        X  = X[ind_list,...]
        y = y[ind_list]
        for count in range(len(y)):
            yield (X[count], y[count])
        
        
def callable_gen(_gen):
        def gen():
            for x,y in _gen:
                 yield x,y
        return gen
    
    
def transform_labels(y):
    y_new = []
    for i in y:
        y_new.append(labels[i])
    return np.array(y_new)


def load_dataset_legacy(data_folder, verbose=True):
    X_train = np.load(data_folder + 'X_train.npy')
    y_train = np.load(data_folder + 'Y_train.npy', allow_pickle=True)
    y_train = transform_labels(y_train)
    
    X_test = np.load(data_folder + 'X_test.npy')
    y_test = np.load(data_folder + 'Y_test.npy', allow_pickle=True)
    y_test = transform_labels(y_test)
    
    if verbose:
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")

        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")
    return X_train, y_train, X_test, y_test


def load_kinetics(config, fold=0):
    
    X_train = np.load('/media/Datasets/kinetics/train_data_joint.npy') #[...,0] # get first pose
    X_train = np.moveaxis(X_train,1,3)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], -1)
    X_train = X_train[:,::config['SUBSAMPLE'],:]
    y_train = np.load('/media/Datasets/kinetics/train_label.pkl', allow_pickle=True)
    y_train = np.transpose(np.array(y_train))[:,1].astype('float32') # get class only
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=config['VAL_SIZE'],
                                                      random_state=config['SEEDS'][fold],
                                                      stratify=y_train)
    
    X_test = np.load('/media/Datasets/kinetics/val_data_joint.npy') #[...,0] # get first pose
    X_test = np.moveaxis(X_test,1,3)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], -1)
    X_test = X_test[:,::config['SUBSAMPLE'],:]
    y_test = np.load('/media/Datasets/kinetics/val_label.pkl', allow_pickle=True)
    y_test = np.transpose(np.array(y_test))[:,1].astype('float32') # get class only
    
    train_gen = callable_gen(kinetics_generator(X_train, y_train, config['BATCH_SIZE']))
    val_gen = callable_gen(kinetics_generator(X_val, y_val, config['BATCH_SIZE']))
    test_gen = callable_gen(kinetics_generator(X_test, y_test, config['BATCH_SIZE']))
    
    return train_gen, val_gen, test_gen, len(y_train), len(y_test)