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
from utils import load_data, plot
from yolomodel import *
from datetime import datetime
from torchmetrics import *


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
parser = argparse.ArgumentParser(description='Process some input')
parser.add_argument('--data', default='data', type=str, help='Dataset path', required=False)   
parser.add_argument('--train','-train', action='store_true', help='Run a training') 
parser.add_argument('--test', '-test', action='store_true', help='Run a test') 
parser.add_argument('--model', default=None, type=str, help='Model path', required=False)   
parser.add_argument('--output', default=None, type=str, help='npy path to plot loss ', required=False)
args = parser.parse_args()
dataset_folder = args.data

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

  if args.model != None: 
    rnn = LSTM(n_joints, n_hidden, n_categories, n_layer)
    model_file_path = args.model
    rnn.load_state_dict(torch.load(model_file_path))
    rnn.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rnn = rnn.to(device)
  else:
    rnn = LSTM(n_joints,n_hidden,n_categories,n_layer).to(device)
  print("model loaded")

  def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return LABELS[category_i], category_i
  
  def accuracy(flag):
      if flag == 'train':
          n = n_data_size_train
      elif flag == 'test':
          n = n_data_size_test
      with torch.no_grad():
          right = 0
          for i in range(n):
              category_tensor, inputs = randomTrainingExampleBatch(1,flag,i)
              category = LABELS[int(category_tensor[0])]
              inputs = inputs.to(device)
              output = rnn(inputs)
              guess, guess_i = categoryFromOutput(output)
              category_i = LABELS.index(category)
              # if flag == 'test':
              #     print(f"guess: {guess}, category: {category}")
              if category_i == guess_i:
                  right+=1
      return right/n

  if args.model == None: 
    print("start training")

    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.0005
    optimizer = optim.SGD(rnn.parameters(),lr=learning_rate,momentum=0.9)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)

    n_iters = 22111
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
    accuracy_ = Accuracy(task = 'multiclass', num_classes = n_categories)

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
            guess, guess_i = categoryFromOutput(output)
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
        
        print(input_sequence_val.size())
        print(output_val.topk(1)[1])
        # print(categoryFromOutput(output_val))
        print(accuracy_(output_val.topk(1)[0], category_tensor_val))
        break
    torch.save(rnn.state_dict(),f'result/{current_time}final.pkl')
    print("Model saved")

  print(f"test accuracy: {accuracy('test')}")
  print(f"train accuracy: {accuracy('train')}")
  # print(f'loss: {all_losses}')
  # plt.figure()
  # plt.plot(all_losses)

  # Keep track of correct guesses in a confusion matrix
  confusion = torch.zeros(n_categories, n_categories)
  n_confusion = n_data_size_test
  precision = np.zeros(n_categories)
  recall = np.zeros(n_categories)
  f1 = np.zeros(n_categories)

  # Go through a bunch of examples and record which are correctly guessed
  for i in range(n_confusion):
      category_tensor, inputs = randomTrainingExampleBatch(1,'test',i)
      # print(f"input: {inputs}")
      category = LABELS[int(category_tensor[0])]
      inputs = inputs.to(device)
      output = rnn(inputs)
      guess, guess_i = categoryFromOutput(output)
      category_i = LABELS.index(category)
      confusion[category_i][guess_i] += 1

  # Normalize by dividing every row by its sum
  for i in range(n_categories):
      confusion[i] = confusion[i] / confusion[i].sum()

  # Print confusion matrix

  with open(f'result/{current_time}loss_conf.npy', 'wb') as f:
    np.save(f, all_losses)
    print("loss saved")
    np.save(f, val_losses)
    print("val loss saved")
    np.save(f, accuracy_train)
    print("accuracy train saved")
    np.save(f, accuracy_val)
    print("accuracy val saved")
    np.save(f, confusion.numpy())
    print("confusion matrix saved")

  # print(confusion.numpy())
  # fig = plt.figure()
  # ax = fig.add_subplot(111)
  # cax = ax.matshow(confusion.numpy())
  # fig.colorbar(cax)

  # # Set up axes
  # ax.set_xticklabels([''] + LABELS, rotation=90)
  # ax.set_yticklabels([''] + LABELS)

  # # Force label at every tick
  # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  # ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

  # # sphinx_gallery_thumbnail_number = 2
  # plt.show()

  for i in range(n_categories):
      true_positives = confusion[i, i]
      false_positives = confusion[:, i].sum() - true_positives
      false_negatives = confusion[i, :].sum() - true_positives

      # Calculate precision, recall, and F1 for the current category
      precision[i] = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
      recall[i] = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0

      # Calculate F1 score
      f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) != 0 else 0

  # Print or use the average F1 score
  average_f1 = np.mean(f1)
  print(f"Average F1 Score: {average_f1}")
