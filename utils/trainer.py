import torch
import torch.nn as nn, optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from utils.transformer import TransformerEncoder, PatchClassEmbedding
from utils.data import load_X, load_y
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import numpy as np


class Model(nn.Module):
    def __init__(self, transformer, config, d_model, mlp_head_size):
        super(Model, self).__init__()
        self.config = config
        self.d_model = d_model
        self.mlp_head_size = mlp_head_size
        self.transformer = transformer

        self.dense = nn.Linear(self.config[self.config['DATASET']]['FRAMES'] // self.config['SUBSAMPLE'] * self.config[self.config['DATASET']]['KEYPOINTS'] * self.config['CHANNELS'], self.d_model)
        self.patch_class_embedding = PatchClassEmbedding(self.d_model, self.config[self.config['DATASET']]['FRAMES'] // self.config['SUBSAMPLE'])
        self.dense_head = nn.Linear(self.d_model, self.mlp_head_size)
        self.output = nn.Linear(self.mlp_head_size, self.config[self.config['DATASET']]['CLASSES'])

    def forward(self, inputs):
        x = self.dense(inputs)
        x = self.patch_class_embedding(x)
        x = self.transformer(x)
        x = x[:, 0, :]
        x = self.dense_head(x)
        outputs = self.output(x)
        return outputs

class Trainer:
    def __init__(self, config, logger, split=1, fold=0):
        self.config = config
        self.logger = logger
        self.split = split
        self.fold = fold
        self.trial = None
        self.bin_path = self.config['MODEL_DIR']
        
        self.model_size = self.config['MODEL_SIZE']
        self.n_heads = self.config[self.model_size]['N_HEADS']
        self.n_layers = self.config[self.model_size]['N_LAYERS']
        self.embed_dim = self.config[self.model_size]['EMBED_DIM']
        self.dropout = self.config[self.model_size]['DROPOUT']
        self.mlp_head_size = self.config[self.model_size]['MLP']
        self.activation = F.gelu
        self.d_model = 64 * self.n_heads
        self.d_ff = self.d_model * 4
        self.pos_emb = self.config['POS_EMB']
    
    def get_model(self):
        transformer = TransformerEncoder(self.d_model, self.n_heads, self.d_ff, self.dropout, self.activation, self.n_layers)
        self.model = Model(transformer, self.config, self.d_model, self.mlp_head_size)

        self.train_steps = np.ceil(float(self.train_len)/self.config['BATCH_SIZE'])
        self.test_steps = np.ceil(float(self.test_len)/self.config['BATCH_SIZE'])
        self.name_model_bin = f"{self.config['MODEL_NAME']}_{self.config['MODEL_SIZE']}_{self.split}_{self.fold}.pt"

        return
    
    
    def get_data(self):
        X_train, y_train, X_test, y_test = load_X(self.config['X_TRAIN']), load_y(self.config['Y_TRAIN']), load_X(self.config['X_TEST']), load_y(self.config['Y_TEST'])
        self.train_len = len(y_train)
        self.test_len = len(y_test)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                            test_size=self.config['VAL_SIZE'],
                                                            random_state=self.config['SEEDS'][self.fold],
                                                            stratify=y_train)
            
        X_train, y_train = torch.tensor(X_train), torch.tensor(y_train)
        X_val, y_val = torch.tensor(X_val), torch.tensor(y_val)
        X_test, y_test = torch.tensor(X_test), torch.tensor(y_test)

        self.ds_train = DataLoader(TensorDataset(X_train, y_train), batch_size=self.config['BATCH_SIZE'])
        self.ds_val = DataLoader(TensorDataset(X_val, y_val), batch_size=self.config['BATCH_SIZE'])
        self.ds_test = DataLoader(TensorDataset(X_test, y_test), batch_size=self.config['BATCH_SIZE'])

    def do_benchmark(self):
        for split in range(1, self.config['SPLITS']+1):      
            self.logger.save_log(f"----- Start Split {split} ----\n")
            self.split = split
            
            acc_list = []
            bal_acc_list = []

            for fold in range(self.config['FOLDS']):
                self.logger.save_log(f"- Fold {fold+1}")
                self.fold = fold
                
                acc, bal_acc = self.do_training()

                acc_list.append(acc)
                bal_acc_list.append(bal_acc)
                
            np.save(self.config['RESULTS_DIR'] + self.config['MODEL_NAME'] + '_' + self.config['DATASET'] + f'_{split}_accuracy.npy', acc_list)
            np.save(self.config['RESULTS_DIR'] + self.config['MODEL_NAME'] + '_' + self.config['DATASET'] + f'_{split}_balanced_accuracy.npy', bal_acc_list)

            self.logger.save_log(f"---- Split {split} ----")
            self.logger.save_log(f"Accuracy mean: {np.mean(acc_list)}")
            self.logger.save_log(f"Accuracy std: {np.std(acc_list)}")
            self.logger.save_log(f"Balanced Accuracy mean: {np.mean(bal_acc_list)}")
            self.logger.save_log(f"Balanced Accuracy std: {np.std(bal_acc_list)}")
            
        
    def do_training(self):
        self.get_data()
        self.get_model()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['LR'])

        best_accuracy = 0.0
        for epoch in range(self.config['N_EPOCHS']):
            self.model.train()
            for inputs, targets in self.ds_train:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            accuracy_test, balanced_accuracy = self.evaluate()
            if accuracy_test > best_accuracy:
                best_accuracy = accuracy_test
                torch.save(self.model.state_dict(), self.bin_path+self.name_model_bin)

        return accuracy_test, balanced_accuracy
    
    def evaluate(self, weights=None):
        if weights is not None:
            self.model.load_state_dict(torch.load(self.bin_path+self.name_model_bin))
        else:
            self.model.load_state_dict(torch.load(self.config['WEIGHTS']))

        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in self.ds_test:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy_test = correct / total
        X, y = zip(*self.ds_test)

        outputs = self.model(X)
        _, y_pred = torch.max(outputs, 1)
        balanced_accuracy = balanced_accuracy_score(y.numpy(), y_pred.numpy())

        text = f"Accuracy Test: {accuracy_test} <> Balanced Accuracy: {balanced_accuracy}\n"
        self.logger.save_log(text)
        return accuracy_test, balanced_accuracy
