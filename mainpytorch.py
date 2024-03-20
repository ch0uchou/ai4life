import numpy as np
import torch.nn as nn
import torch
import random
import torch.optim as optim
import time
import math

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

LABELS = [
  "russian twist",
  "tricep dips",
  "t bar row",
  "squat",
  "shoulder press",
  "romanian deadlift",
  "push-up",
  "plank",
  "leg extension",
  "leg raises",
  "lat pulldown",
  "incline bench press",
  "tricep pushdown",
  "pull up",
  "lateral raise",
  "hammer curl",
  "decline bench press",
  "hip thrust",
  "bench press",
  "chest fly machine",
  "deadlift",
  "barbell biceps curl"
]

X_train_path = file_path_trainx
X_test_path = file_path_testx

y_train_path = file_path_trainy
y_test_path = file_path_testy

n_steps = 32 # 32 timesteps per series
n_categories = len(LABELS)
label_number = 6
split_ratio = 0.8
# Load the networks inputs

def load_X(X_path):
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
    return y_

X_train = load_X(X_train_path)
X_test = load_X(X_test_path)

y_train = load_y(y_train_path)
y_test = load_y(y_test_path)

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

class LSTM(nn.Module):
  def __init__(self,input_dim,hidden_dim,output_dim,layer_num):
    super(LSTM,self).__init__()
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.lstm = torch.nn.LSTM(input_dim,hidden_dim,layer_num,batch_first=True)
    self.fc = torch.nn.Linear(hidden_dim,output_dim)
    self.bn = nn.BatchNorm1d(32)

  def forward(self,inputs):
    x = self.bn(inputs)
    lstm_out,(hn,cn) = self.lstm(x)
    out = self.fc(lstm_out[:,-1,:])
    return out

def randomTrainingExampleBatch(batch_size,flag,num=-1):
  if flag == 'train':
    X = tensor_X_train
    y = tensor_y_train
    data_size = n_data_size_train
  elif flag == 'test':
    X = tensor_X_test
    y = tensor_y_test
    data_size = n_data_size_test
  if num == -1:
    ran_num = random.randint(0,data_size-batch_size)
  else:
    ran_num = num
  pose_sequence_tensor = X[ran_num:(ran_num+batch_size)]
  pose_sequence_tensor = pose_sequence_tensor
  category_tensor = y[ran_num:ran_num+batch_size,:]
  return category_tensor.long(),pose_sequence_tensor

n_hidden = 128
n_joints = 17*2
n_categories = 22
n_layer = 3
rnn = LSTM(n_joints,n_hidden,n_categories,n_layer).to(device)
