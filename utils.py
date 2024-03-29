import numpy as np
from sklearn.utils import shuffle
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torchmetrics import *
from model import LSTM, TransformerModel


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

def load_data(X_path, y_path, train_flag = True, n_frame = 32, shuffle_flag=True, device='cuda'):
  X_= load_X(X_path, n_frame)
  y_= load_y(y_path)
  
  if train_flag == True:
    if shuffle_flag:
        X_, y_= shuffle(X_, y_)
        
  tensor_X= torch.from_numpy(X_).to(device)
  print('data_size:',tensor_X.size())
  tensor_y= torch.from_numpy(y_).to(device)
  print('label_size:',tensor_y.size())
  n_data_size= tensor_X.size()[0]
  print('n_data_size:',n_data_size)
  return tensor_X, tensor_y, n_data_size

def plot_loss_acc(file_path, LABELS):
    with open(f'{file_path}', 'rb') as f:
        train_losses = np.load(f)
        val_losses = np.load(f)
    print(np.argmin(val_losses))
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(train_losses)
    plt.title('Train Losses')
    plt.subplot(2, 1, 2)
    plt.plot(val_losses)
    plt.title('Validation Losses')
    plt.show()
    

def plot_confusion_matrix(file_path, LABELS):
    with open(f'{file_path}', 'rb') as f:
        confusion = np.load(f)
    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion)
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + LABELS, rotation=90)
    ax.set_yticklabels([''] + LABELS)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()

def accuracy(pred, target, n_categories, device='cuda'):
    accuracy_ = Accuracy(task = 'multiclass', num_classes = n_categories).to(device)
    return accuracy_(torch.reshape(pred.topk(1)[1],(-1,)), target).cpu()

def confusion_matrix(pred, target, n_categories, device='cuda'):
    confmat = ConfusionMatrix(task="multiclass", num_classes=n_categories, normalize='all').to(device)
    return confmat(pred, target).cpu().numpy()

def f1_score(pred, target, n_categories, device='cuda'):
    f1 = F1Score(task="multiclass",num_classes=n_categories).to(device)
    return f1(torch.reshape(pred.topk(1)[1],(-1,)), target).cpu()

def get_output_from_model(model, X, y, device='cuda'):
    category_tensor, input_sequence = y.long(), X
    input_sequence = input_sequence.to(device)
    category_tensor = category_tensor.to(device)
    category_tensor = torch.squeeze(category_tensor)
    output = model(input_sequence)
    return output, category_tensor

def load_model(file_path, n_joints, n_hidden, n_categories, n_layer, device='cuda'):
    if file_path == None:
        # rnn = LSTM(n_joints,n_hidden,n_categories,n_layer).to(device)
        rnn = TransformerModel(n_joints,n_hidden,n_categories,n_layer).to(device)
    else:
        # rnn = LSTM(n_joints, n_hidden, n_categories, n_layer)
        rnn = TransformerModel(n_joints,n_hidden,n_categories,n_layer)
        model_file_path = file_path
        rnn.load_state_dict(torch.load(model_file_path))
        rnn.eval()
        rnn = rnn.to(device)
    return rnn