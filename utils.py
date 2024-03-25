import numpy as np
from sklearn.utils import shuffle
import torch

# Load the networks inputs

def load_X(X_path, n_steps):
    file = open(X_path, 'r')
    X_ = np.array(
        [elem for elem in [
            row.split(',') for row in file
        ]],
        dtype=np.float32
    )
    file.close()
    blocks = int(len(X_) / n_steps)

    X_ = np.array(np.split(X_,blocks))

    return X_

# Load the networks outputs
def load_y(y_path):
    file = open(y_path, 'r')
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()

    # for 0-based indexing
    return y_ -1

def load_data(X_train_path, y_train_path, X_test_path, y_test_path, n_frame = 32, shuffle_flag=True):
  X_train = load_X(X_train_path, n_frame)
  X_test = load_X(X_test_path, n_frame)

  y_train = load_y(y_train_path)
  y_test = load_y(y_test_path)
  
  if shuffle_flag:
    X_train, y_train = shuffle(X_train, y_train)

  tensor_X_test = torch.from_numpy(X_test)
  print('test_data_size:',tensor_X_test.size())
  tensor_y_test = torch.from_numpy(y_test)
  print('test_label_size:',tensor_y_test.size())
  n_data_size_test = tensor_X_test.size()[0]
  print('n_data_size_test:',n_data_size_test)

  tensor_X_train = torch.from_numpy(X_train)
  print('train_data_size:',tensor_X_train.size())
  tensor_y_train = torch.from_numpy(y_train)
  print('train_label_size:',tensor_y_train.size())
  n_data_size_train = tensor_X_train.size()[0]
  print('n_data_size_train:',n_data_size_train)
  return tensor_X_train, tensor_y_train, tensor_X_test, tensor_y_test, n_data_size_train, n_data_size_test