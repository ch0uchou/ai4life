import numpy as np
import torch.nn as nn
import random
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import math
import random
import torch.optim as optim
import time
import math
import matplotlib.ticker as ticker
import argparse
from utils import load_data, plot, accuracy, confusion_matrix
from yolomodel import *
from datetime import datetime


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
parser = argparse.ArgumentParser(description='Process some input')
parser.add_argument('--data', default='data', type=str, help='Dataset path', required=False)   
parser.add_argument('--train','-train', action='store_true', help='Run a training') 
parser.add_argument('--test', '-test', action='store_true', help='Run a test') 
parser.add_argument('--model', default=None, type=str, help='Model path', required=False)   
parser.add_argument('--output', default=None, type=str, help='npy path to plot loss ', required=False)
args = parser.parse_args()
dataset_folder = args.data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

if args.output != None:
 plot(args.output, LABELS)
else:
  if args.train:
    X_train_path, y_train_path, X_test_path, y_test_path = train(dataset_folder, LABELS, current_time)
  elif args.test:
    X_train_path, y_train_path, X_test_path, y_test_path = test(dataset_folder, LABELS, current_time)
  else:
    X_train_path = "20240326-140332dataX_train.txt"
    y_train_path = "20240326-140332dataY_train.txt"
    X_test_path = "20240326-140332dataX_test.txt"
    y_test_path = "20240326-140332dataY_test.txt"

  n_steps = 32 # 32 timesteps per series
  n_categories = len(LABELS)
  label_number = 6
  split_ratio = 0.8
  # Load the networks inputs

  tensor_X_train, tensor_y_train, tensor_X_test, tensor_y_test, n_data_size_train, n_data_size_test = load_data(X_train_path, y_train_path, X_test_path, y_test_path, n_steps, shuffle_flag=True)

  class LSTM(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,layer_num):
      super(LSTM,self).__init__()
      self.hidden_dim = hidden_dim
      self.output_dim = output_dim
      self.lstm = torch.nn.LSTM(input_dim,hidden_dim,layer_num,batch_first=True)
      self.fc = torch.nn.Linear(hidden_dim,output_dim)
      self.bn = nn.BatchNorm1d(32)

      # self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3)
      # self.relu = nn.ReLU()
      # self.lstm1 = nn.LSTM(input_size=64, hidden_size=hidden_dim, num_layers=layer_num, batch_first=True)
      # self.lstm2 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=layer_num, batch_first=True)
      # self.fc = nn.Linear(hidden_dim, n_categories)


    def forward(self,inputs):
      x = self.bn(inputs)
      lstm_out,_ = self.lstm(x)
      out = self.fc(lstm_out[:,-1,:])
      return out
      # x = inputs.permute(0, 2, 1)  # Reshape to (batch_size, 34, 32) for 1D conv
      # x = self.conv1d(x)
      # x = self.relu(x)
      
      # # Reshape back to (batch_size, 32, 64) for LSTM
      # x = x.permute(0, 2, 1)
      # # LSTM expects input of shape (batch_size, seq_len, input_size)
      
      # # First LSTM layer
      # lstm_out1, _ = self.lstm1(x)
      
      # # Second LSTM layer
      # lstm_out2, _ = self.lstm2(lstm_out1)
      
      # # Get output from the last time step
      # lstm_out = lstm_out2[:, -1, :]
      
      # # Fully connected layer
      # output = self.fc(lstm_out)
      # return output

  n_hidden = 128
  n_joints = 17*2
  n_categories = 22
  n_layer = 3

  if args.model != None: 
    rnn = LSTM(n_joints, n_hidden, n_categories, n_layer)
    model_file_path = args.model
    rnn.load_state_dict(torch.load(model_file_path))
    rnn.eval()
    rnn = rnn.to(device)
  else:
    rnn = LSTM(n_joints,n_hidden,n_categories,n_layer).to(device)
  print("model loaded")
  

  if args.model == None: 
    print("start training")

    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.0005
    optimizer = optim.SGD(rnn.parameters(),lr=learning_rate,momentum=0.9)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)

    n_iters = 2000
    #n_iters = 60000
    print_every = 1000
    plot_every = 100
    batch_size = 128

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    val_losses = []
    accuracy_train = []
    accuracy_val = []
    def timeSince(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    start = time.time()

    for iter in range(1, n_iters + 1):
        category_tensor, input_sequence = tensor_y_train.long(), tensor_X_train
        input_sequence = input_sequence.to(device)
        category_tensor = category_tensor.to(device)
        category_tensor = torch.squeeze(category_tensor)

        optimizer.zero_grad()
        output = rnn(input_sequence)
        loss = criterion(output, category_tensor)
        loss.backward()
        optimizer.step()
        #scheduler.step()
        current_loss += loss.item()

        category = LABELS[int(category_tensor[0])]

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess = LABELS[torch.reshape(output.topk(1)[1],(-1,))[0].item()]
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f  / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, guess, correct))

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
          all_losses.append(current_loss / plot_every)
          current_loss = 0    
          
        category_tensor_val, input_sequence_val = tensor_y_test.long(), tensor_X_test
        input_sequence_val = input_sequence_val.to(device)
        category_tensor_val = category_tensor_val.to(device)
        category_tensor_val = torch.squeeze(category_tensor_val)
        output_val = rnn(input_sequence_val)        
        loss_val = criterion(output_val, category_tensor_val)
        val_losses.append(loss_val.item())
        
        accuracy_train.append(accuracy(output, category_tensor))
        accuracy_val.append(accuracy(output_val, category_tensor_val))
    torch.save(rnn.state_dict(),f'result/{current_time}final.pkl')
    print("Model saved")

  # confusion = confusion_matrix()

  with open(f'result/{current_time}loss_conf.npy', 'wb') as f:
    np.save(f, all_losses)
    print("loss saved")
    np.save(f, val_losses)
    print("val loss saved")
    np.save(f, accuracy_train)
    print("accuracy train saved")
    np.save(f, accuracy_val)
    print("accuracy val saved")
    # np.save(f, confusion.numpy())
    # print("confusion matrix saved")
