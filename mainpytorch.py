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
from utils import *
from yolomodel import *
from datetime import datetime
from model import LSTM


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
parser = argparse.ArgumentParser(description='Process some input')
parser.add_argument('--data', default='data', type=str, help='Dataset path', required=False)   
parser.add_argument('--train','-train', action='store_true', help='Run a training') 
parser.add_argument('--test', '-test', action='store_true', help='Run a test') 
parser.add_argument('--model', default=None, type=str, help='Model path', required=False)   
parser.add_argument('--plot', default=None, type=str, help='npy path to plot loss ', required=False)
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

if args.plot != None:
  # plot_loss_acc(args.plot, LABELS)
  plot_confusion_matrix(args.plot, LABELS)
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

    n_iters = 100000
    #n_iters = 60000
    print_every = 1000
    batch_size = 128

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    def timeSince(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    start = time.time()

    for iter in range(1, n_iters + 1):
        output, category_tensor = get_output_from_model(rnn, tensor_X_train, tensor_y_train.long())
        optimizer.zero_grad()
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

        #get loss of train set every plot_every iterations
        all_losses.append(loss.item())  
      
        #get loss of val set every plot_every iterations
        output_val, category_tensor_val = get_output_from_model(rnn, tensor_X_test, tensor_y_test.long())
        loss_val = criterion(output_val, category_tensor_val)
        val_losses.append(loss_val.item())
        
        train_accuracies.append(accuracy(output, category_tensor, n_categories))
        val_accuracy = accuracy(output_val, category_tensor_val, n_categories)
        if (val_accuracy > (max(val_accuracies) if len(val_accuracies) > 0 else 0)):
          torch.save(rnn.state_dict(),f'result/{current_time}final.pkl')
          print(f"find accuracy {val_accuracy} > {(max(val_accuracies) if len(val_accuracies) > 0 else 0)} save model")
        val_accuracies.append(val_accuracy)
    with open(f'result/{current_time}loss.npy', 'wb') as f:
      np.save(f, all_losses)
      print("loss saved")
      np.save(f, val_losses)
      print("val loss saved")
      np.save(f, train_accuracies)
      print("accuracy train saved")
      np.save(f, val_accuracies)
      print("accuracy val saved")

  output_test, category_tensor_test = get_output_from_model(rnn, tensor_X_test, tensor_y_test.long())
  print(f'test accuracy: {accuracy(output_test, category_tensor_test, n_categories).item()}')
  confusion = confusion_matrix(output_test, category_tensor_test, n_categories)
  with open(f'result/{current_time}confusion_matrix.npy', 'wb') as f:
      np.save(f, confusion)
      print("confusion matrix saved")
 