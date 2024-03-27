import numpy as np
from sklearn.utils import shuffle
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torchmetrics import *


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

def load_data(X_train_path, y_train_path, X_test_path, y_test_path, n_frame = 32, shuffle_flag=True, device='cuda'):
  X_train = load_X(X_train_path, n_frame)
  X_test = load_X(X_test_path, n_frame)

  y_train = load_y(y_train_path)
  y_test = load_y(y_test_path)
  
  if shuffle_flag:
    X_train, y_train = shuffle(X_train, y_train)

  tensor_X_test = torch.from_numpy(X_test).to(device)
  print('test_data_size:',tensor_X_test.size())
  tensor_y_test = torch.from_numpy(y_test).to(device)
  print('test_label_size:',tensor_y_test.size())
  n_data_size_test = tensor_X_test.size()[0]
  print('n_data_size_test:',n_data_size_test)

  tensor_X_train = torch.from_numpy(X_train).to(device)
  print('train_data_size:',tensor_X_train.size())
  tensor_y_train = torch.from_numpy(y_train).to(device)
  print('train_label_size:',tensor_y_train.size())
  n_data_size_train = tensor_X_train.size()[0]
  print('n_data_size_train:',n_data_size_train)
  return tensor_X_train, tensor_y_train, tensor_X_test, tensor_y_test, n_data_size_train, n_data_size_test

def plot(file_path, LABELS):
    with open(f'{file_path}', 'rb') as f:
        train_losses = np.load(f)
        val_losses = np.load(f)
        train_accuracies = np.load(f)
        val_accuracies = np.load(f)
        confusion = np.load(f)
    print(np.argmin(val_losses))
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(train_losses)
    plt.title('Train Losses')
    plt.subplot(2, 1, 2)
    plt.plot(val_losses)
    plt.title('Validation Losses')
    plt.subplot(2, 1, 1)
    plt.plot(train_accuracies)
    plt.title('Train Accuracies')
    plt.subplot(2, 1, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracies')
    
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
    confmat = ConfusionMatrix(task="multiclass", num_classes=n_categories).to(device)
    confusion = confmat()
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()
    return confusion

def get_output_from_model(model, X, y, n_categories, device='cuda'):
    category_tensor, input_sequence = y.long(), X
    input_sequence = input_sequence.to(device)
    category_tensor = category_tensor.to(device)
    category_tensor = torch.squeeze(category_tensor)
    output = model(input_sequence)
    return output