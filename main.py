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
import copy



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
parser = argparse.ArgumentParser(description='Process some input')
parser.add_argument('--data', default='data', type=str, help='Dataset path', required=False)   
parser.add_argument('--train','-train', action='store_true', help='Run a training') 
parser.add_argument('--test', '-test', action='store_true', help='Run a test')
parser.add_argument('--testfold', default=None, type=str, help='Test folder path', required=False)   
parser.add_argument('--txt', '-txt', action='store_true', help='Run on txt file') 
parser.add_argument('--model', default=None, type=str, help='Model path', required=False)   
parser.add_argument('--plot', default=None, type=str, help='npy path to plot loss ', required=False)
args = parser.parse_args()
dataset_folder = args.data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.isdir("result"):
    os.makedirs("result")
  
LABELS = {
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
  "barbell biceps curl",
}
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
n_steps = 32 # 32 timesteps per series
n_categories = len(LABELS)
label_number = 6
split_ratio = 0.8
# Load the networks inputs
n_hidden = 128
n_joints = 17*2
n_categories = 22
n_layer = 3

def trainning(rnn, X_train_path, y_train_path, X_val_path, y_val_path, n_steps):
    tensor_X_train, tensor_y_train, n_data_size_train = load_data(X_train_path, y_train_path, n_frame=n_steps, shuffle_flag=True)
    tensor_X_val, tensor_y_val, n_data_size_val = load_data(X_val_path, y_val_path, n_frame=n_steps, shuffle_flag=True, train_flag=False)
    print("start training")

    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.0005
    optimizer = optim.SGD(rnn.parameters(),lr=learning_rate,momentum=0.9)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)

    n_iters = 60000
    #n_iters = 60000
    print_every = 1000
    plot_every = 10

    # Keep track of losses for plotting
    current_loss = 0
    current_loss_val = 0
    all_losses = []
    val_losses = []
    min_val_loss = 1000000
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
        print(category_tensor[0])
        category = LABELS[int(category_tensor[0].item())]
      
        #get loss of val set every plot_every iterations
        output_val, category_tensor_val = get_output_from_model(rnn, tensor_X_val, tensor_y_val.long())
        loss_val = criterion(output_val, category_tensor_val)
        current_loss_val += loss_val.item() 
        
        #plot loss
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0
            val_losses.append(current_loss_val / plot_every)
            if ((current_loss_val / plot_every) <= min_val_loss):
              torch.save(rnn.state_dict(),f'result/{current_time}final.pkl')
              min_val_loss = current_loss_val / plot_every
            
        # Print iter number, loss, name and guess
        if iter % print_every == 0: 
            guess = LABELS[torch.reshape(output.topk(1)[1],(-1,))[0].item()]
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) loss: %.4f val_loss: %.4f / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, loss_val, guess, correct))
    print("best model saved")
    with open(f'result/{current_time}loss.npy', 'wb') as f:
      np.save(f, all_losses)
      print(f"loss saved")
      np.save(f, val_losses)
      print(f"val loss saved {min_val_loss}")


def test(rnn, tensor_X_test, tensor_y_test, n_categories, testfold=None):
  if testfold == None:
    output_test, category_tensor_test = get_output_from_model(rnn, tensor_X_test, tensor_y_test.long())
    guess = output_test
    true = torch.reshape(category_tensor_test,(-1,))
    print(f'Guess: {guess}')
    print(f'True: {true}')
    confusion = confusion_matrix(output_test, category_tensor_test, n_categories)
    f1 = f1_score(output_test, category_tensor_test, n_categories)
    print(f'f1 score: {f1}')
    with open(f'result/{current_time}confusion_matrix.npy', 'wb') as f:
      np.save(f, confusion)
      print("confusion matrix saved")
      np.save(f, f1.numpy())
      print("f1 score saved")
  else:
    tensor_X_test = tensor_X_test.to(device)
    output = rnn(tensor_X_test)
    print(f'{output.topk(1)[1]}')
    # print(f'{tensor_y_test}')

if args.plot != None:
  plot_loss_acc(args.plot, LABELS)
  # plot_confusion_matrix(args.plot, LABELS)
elif args.txt:
  X_train_path = "20240401-193433dataX_train.txt"
  y_train_path = "20240401-193433dataY_train.txt"
  X_val_path = "20240401-193433dataX_valid.txt"
  y_val_path = "20240401-193433dataY_train.txt"
  X_test_path = "20240401-203835dataX.txt"
  y_test_path = "20240401-203835dataY.txt"
else:
  if args.train:
    X_train_path, y_train_path, X_val_path, y_val_path = get_trainset(dataset_folder, LABELS, current_time)
  elif args.test:
    X_test_path, y_test_path = get_testset(dataset_folder, LABELS, current_time, args.testfold)


if args.train: 
  rnn = load_model(None, n_joints, n_hidden, n_categories, n_layer)
  print("model loaded")
  trainning(rnn, X_train_path, y_train_path, X_val_path, y_val_path, n_steps)

if args.test:
  if args.model == None:
      print("Please provide a model path")
      exit()
  rnn = load_model(args.model, n_joints, n_hidden, n_categories, n_layer)
  print("model loaded")
  if args.testfold == None:
    tensor_X_test, tensor_y_test, n_data_size_test = load_data(X_test_path, y_test_path, n_frame=n_steps, shuffle_flag=False, train_flag=False)
    test(rnn, tensor_X_test, tensor_y_test, n_categories)
  else:
    tensor_X_test = load_X(X_test_path, n_steps=n_steps)
    tensor_X_test = torch.from_numpy(tensor_X_test).to(device)
    file = open(y_test_path, 'r')
    tensor_y_test = np.array(
      [elem for elem in [
        row.replace('  ', ' ').strip().split(' ') for row in file
      ]]
    )
    file.close()
    test(rnn, tensor_X_test, tensor_y_test, n_categories, args.testfold)