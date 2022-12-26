from torch import nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import Adam

from sklearn.metrics import f1_score, confusion_matrix

import warnings
import math
import numpy as np
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from tqdm import tqdm
from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore")

class Conv1dSame(nn.Conv1d):

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s[0]) - 1) * s[0] + (k[0] - 1) * d[0] + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad = self.calc_same_pad(i=ih, k=self.kernel_size, s=self.stride, d=self.dilation)
        if pad > 0 or pad > 0:
            x = F.pad(
                x, [pad // 2, pad - pad // 2]
            )
        return F.conv1d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

class MaxPool1dSame(nn.MaxPool1d):

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        g = max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)
        return g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad = self.calc_same_pad(i=ih, k=self.kernel_size, s=self.stride, d=self.dilation)
        if pad > 0 or pad > 0:
            x = F.pad(
                x, [pad // 2, pad - pad // 2]
            )
        return F.max_pool1d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
        )


class ExtendedNet(nn.Module):
    def __init__(self, out_features=None, bidirectional=None, bias=None, dropout=None, load_existing=False, existing_path=None, device=None):
        super(ExtendedNet, self).__init__()
        self.relu = nn.ReLU()
        if not load_existing:
            self.dropout_p = dropout
            self.out_features = out_features
            self.bidirectional = bidirectional
            self.bias = True
        else:
            state_dict = self.load_checkpoint(device, existing_path, False)
        self.bias=True


        self.dropout = nn.Dropout(p=self.dropout_p)

        downsample_1 = 1
        downsample_2 = 4
        downsample_3 = 4
        downsample_4 = 4

        self.conv_in = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=16, padding='same', bias=self.bias)
        self.batchnorm_in = nn.BatchNorm1d(num_features=64)

        self.conv_main_1_1 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=16, padding='same', bias=self.bias)
        self.conv_main_1_2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=16, bias=self.bias, stride=downsample_1)
        self.batchnorm_main_1 = nn.BatchNorm1d(num_features=128)
        self.maxpool_skip_1 = nn.MaxPool1d(kernel_size=16, stride=downsample_1)
        self.conv_skip_1 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=16, padding='same', bias=self.bias)

        self.conv_main_2_1 = nn.Conv1d(in_channels=128, out_channels=196, kernel_size=16, padding='same', bias=self.bias)
        self.conv_main_2_2 = nn.Conv1d(in_channels=196, out_channels=196, kernel_size=16, bias=self.bias, stride=downsample_2)
        self.batchnorm_main_2 = nn.BatchNorm1d(num_features=196)
        self.maxpool_skip_2 = nn.MaxPool1d(kernel_size=16, stride=downsample_2)
        self.conv_skip_2 = nn.Conv1d(in_channels=128, out_channels=196, kernel_size=16, padding='same', bias=self.bias)

        self.conv_main_3_1 = nn.Conv1d(in_channels=196, out_channels=256, kernel_size=16, padding='same', bias=self.bias)
        self.conv_main_3_2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=16, bias=self.bias, stride=downsample_3)
        self.batchnorm_main_3 = nn.BatchNorm1d(num_features=256)
        self.maxpool_skip_3 = nn.MaxPool1d(kernel_size=16, stride=downsample_3)
        self.conv_skip_3 = nn.Conv1d(in_channels=196, out_channels=256, kernel_size=16, padding='same', bias=self.bias)

        self.conv_main_4_1 = nn.Conv1d(in_channels=256, out_channels=320, kernel_size=16, padding='same', bias=self.bias)
        self.conv_main_4_2 = nn.Conv1d(in_channels=320, out_channels=320, kernel_size=16, bias=self.bias, stride=downsample_4)
        self.batchnorm_main_4 = nn.BatchNorm1d(num_features=320)
        self.maxpool_skip_4 = nn.MaxPool1d(kernel_size=16, stride=downsample_4)
        self.conv_skip_4 = nn.Conv1d(in_channels=256, out_channels=320, kernel_size=16, padding='same', bias=self.bias)

        self.batchnorm_final = nn.BatchNorm1d(num_features=320)
        if load_existing: self.out_features = int(self.out_features)
        self.bi_LSTM = nn.LSTM(input_size=320, hidden_size=self.out_features, num_layers=3, dropout=0.2, bidirectional=self.bidirectional, batch_first=True)
        if self.bidirectional: out_features = self.out_features *2
        else: out_features = self.out_features

        self.fc_1 = nn.Linear(in_features=out_features, out_features=200)
        self.fc_2 = nn.Linear(in_features=200, out_features=200)
        self.fc = nn.Linear(in_features=200, out_features=4)

        if load_existing: self.load_state_dict(state_dict)

    def forward(self, x, classify):
        x = self.conv_in(x)
        x = self.relu(self.batchnorm_in(x))

        # RESIDUAL UNIT 1
        y = torch.clone(x)
        y = self.maxpool_skip_1(y)

        y = self.conv_skip_1(y)

        x = self.conv_main_1_1(x)
        x = self.relu(self.batchnorm_main_1(x))
        x = self.dropout(x)

        x = self.conv_main_1_2(x)
        x = x+y
        x = self.relu(self.batchnorm_main_1(x))
        x = self.dropout(x)

        # RESIDUAL UNIT 2
        y = torch.clone(x)
        y = self.maxpool_skip_2(y)
        y = self.conv_skip_2(y)

        x = self.conv_main_2_1(x)
        x = self.relu(self.batchnorm_main_2(x))
        x = self.dropout(x)
        x = self.conv_main_2_2(x)
        x = x + y
        x = self.relu(self.batchnorm_main_2(x))
        x = self.dropout(x)

        # RESIDUAL UNIT 3
        y = torch.clone(x)
        y = self.maxpool_skip_3(y)
        y = self.conv_skip_3(y)

        x = self.conv_main_3_1(x)
        x = self.relu(self.batchnorm_main_3(x))
        x = self.dropout(x)

        x = self.conv_main_3_2(x)
        x = x + y
        x = self.relu(self.batchnorm_main_3(x))
        x = self.dropout(x)

        # RESIDUAL UNIT 4
        y = torch.clone(x)
        y = self.maxpool_skip_4(y)
        y = self.conv_skip_4(y)

        x = self.conv_main_4_1(x)
        x = self.relu(self.batchnorm_main_4(x))
        x = self.dropout(x)

        x = self.conv_main_4_2(x)
        x = x + y
        x = self.relu(self.batchnorm_main_4(x))
        x = self.dropout(x)

        x = torch.permute(x, (0,2,1))
        x, _ = self.bi_LSTM(x)
        x = x[:,-1,:]
        x = self.dropout(self.relu(self.fc_1(x)))
        x = self.dropout(self.relu(self.fc_2(x)))
        if classify:
            x = self.fc(x)
        return x

    def extract_windows(self, X_, y_, window_size, padding):
        x_windows = []
        y_windows = []
        _, weight = np.unique(y_, return_counts=True)
        a = 4
        b = int(( 1+ ((weight[0]/weight[1])-1)*self.imb_weight)*a)
        c = int(( 1+ ((weight[0]/weight[2])-1)*self.imb_weight)*a)
        d = int(( 1+ ((weight[0]/weight[3])-1)*self.imb_weight)*a)
        print(a,b,c,d)


        for x, y in zip(X_, y_):
            if x.shape[0] < window_size:
                if padding == '0':
                    padded = np.pad(x, (window_size - x.shape[0] % window_size, 0))
                    x_windows.append(padded[None, :])
                    y_windows.append(y)
                elif padding == 'wrap':
                    padded = np.pad(x, (window_size - x.shape[0] % window_size, 0), mode='wrap')
                    x_windows.append(padded[None, :])
                    y_windows.append(y)
                else:
                    pass
            else:
                if y == 0:
                    stride = int(window_size / a)
                elif y == 1:
                    stride = int(window_size / b)
                elif y == 2:
                    stride = int(window_size / c)
                elif y == 3:
                    stride = int(window_size / d)
                for i in range(int(np.ceil((x.shape[0] - window_size) / stride)) + 1):
                    x_i = x[i * stride:i * stride + window_size]
                    if x_i.shape[0] == window_size:
                        x_windows.append(x_i[None, :])
                    else:
                        x_windows.append(x[-window_size:][None, :])
                    y_windows.append(y)

        x_windows = np.concatenate(x_windows, axis=0)
        y_windows = np.array(y_windows)

        return x_windows, y_windows

    def get_dataloader(self, X_, y_, train_split, window_size, batch_size, padding):
        X_train, X_val, y_train, y_val = train_test_split(X_, y_, shuffle=True, train_size=train_split)
        X_train, y_train = self.extract_windows(X_train, y_train, window_size, padding)
        X_val, y_val = self.extract_windows(X_val, y_val, window_size, padding)

        _, weight = np.unique(y_train, return_counts=True)
        print('sampled class count:', weight)
        weight = 1 / weight
        weight = weight / np.sum(weight)
        print('class weights:', weight)
        weight_mid = (np.max(weight) - np.min(weight))*0.5+np.min(weight)
        for i in range(weight.shape[0]):
            weight[i] = weight[i] + (weight_mid-weight[i])*self.weight_scale
        print('damped class weights:', weight)
        weight = torch.tensor(weight).float()


        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        return DataLoader(train_dataset, batch_size=batch_size), DataLoader(val_dataset, batch_size=batch_size), weight

    def train_net(self, X, y, train_split, n_epochs, experiment_path, device, lr_step_size, batch_size, window_size, padding, imb_weight, weight_scale, patience):
        self.patience = patience
        self.weight_scale = weight_scale
        self.imb_weight = imb_weight
        self.padding = padding
        self.batch_size = batch_size
        self.window_size = window_size
        train_loader, val_loader, weight = self.get_dataloader(X, y, train_split, self.window_size, batch_size, self.padding)
        self.lr_step_size = lr_step_size
        optimizer = Adam(self.parameters())
        criterion = nn.CrossEntropyLoss(weight=weight).to(device)
        # criterion = nn.CrossEntropyLoss().to(device)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=1/2, patience=patience, verbose=True)

        self.val_loss_list = []
        self.loss_list = []
        self.min_loss = 100000

        self.save_model_path = experiment_path + str(self.out_features)+'_' + str(self.window_size) + '_' + str(self.batch_size) + '_' + str(self.dropout_p) + '_' + str(padding) + '_' + str(self.imb_weight)+'_'+str(self.weight_scale)+'_'+str(self.patience)+'.pth'

        plateau = 0
        for self.epoch in range(n_epochs):
            running_loss = 0.0
            print('epoch', self.epoch)
            self.train()
            for i, data in enumerate(train_loader):
                input, target = data
                optimizer.zero_grad()

                classes_pred = self.forward(input[:,None,:].to(device), True)
                loss = criterion(classes_pred.to(device), target.to(device))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            preds=[]
            targets=[]
            self.eval()
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    input, target = data
                    preds.append(self.forward(input[:,None,:].to(device), True))
                    targets.append(target)
                #     score = f1_score(target.cpu().numpy(), np.argmax(pred.detach().cpu().numpy(), axis=1), average='micro')
                #     score_weight = input.shape[0]
                #     score *= score_weight
                #
                #     metric_sum += score
                #     div += score_weight
                # accuracy = metric_sum / div
                preds = torch.cat(preds)
                targets = torch.cat(targets)
                val_loss = criterion(preds.to(device), targets.to(device)).item()
                accuracy = f1_score(targets.cpu().numpy(), np.argmax(preds.cpu().numpy(), axis=1), average='micro')
                accuracy_mean = f1_score(targets.cpu().numpy(), np.argmax(preds.cpu().numpy(), axis=1), average='macro')
                confusion = confusion_matrix(targets.cpu().numpy(), np.argmax(preds.cpu().numpy(), axis=1))

            self.val_loss_list.append(np.around(val_loss, decimals=4))
            self.loss_list.append(running_loss)


            if val_loss < self.min_loss:
                plateau = 0
                self.min_loss = val_loss
                self.save_checkpoint(self.save_model_path)
            else: plateau += 1


            print(self.loss_list)
            print(self.val_loss_list)
            print('f1:',accuracy)
            print(confusion)
            # print(confusion[0,0]/np.sum(confusion[0,:]), confusion[1,1]/np.sum(confusion[1,:]),
            #       confusion[2,2]/np.sum(confusion[2,:]), confusion[3,3]/np.sum(confusion[3,:]))
            print('each f1:', confusion[0,0]/(confusion[0,0]+0.5*(confusion[0,1]+confusion[0,2]+confusion[0,3]+confusion[1,0]+confusion[2,0]+confusion[3,0])),
                  confusion[1, 1] / (confusion[1, 1] + 0.5 * (confusion[1, 0] + confusion[1, 2] + confusion[1, 3] + confusion[0,1] + confusion[2, 1] + confusion[3, 1])),
                  confusion[2, 2] / (confusion[2, 2] + 0.5 * (confusion[2, 0] + confusion[2, 1] + confusion[2, 3] + confusion[0,2] + confusion[1, 2] + confusion[3, 2])),
                  confusion[3, 3] / (confusion[3, 3] + 0.5 * (confusion[3, 0] + confusion[3, 1] + confusion[3, 2] + confusion[0,3] + confusion[1, 3] + confusion[2, 3])))
            print('f1 mean:', accuracy_mean)

            if plateau == self.patience*3:
                print('stopped at epoch:', self.epoch)
                break

            bad_epochs = scheduler.num_bad_epochs
            scheduler.step(val_loss)
            print('num bad epochs:', scheduler.num_bad_epochs)

            if scheduler.num_bad_epochs == 0 and bad_epochs == self.patience:
                print('network reseting')
                self.load_checkpoint(device, self.save_model_path)


    def save_checkpoint(self, PATH):
        torch.save({'loss_tracker': self.min_loss,
                    'net_state_dict': self.state_dict(),
                    'current_epoch': self.epoch,
                    'dropout_p': self.dropout_p,
                    'out_features': self.out_features,
                    'bidirectional': self.bidirectional,
                    'lr_step_size': self.lr_step_size,
                    'val_loss_list': self.val_loss_list,
                    'loss_list': self.loss_list,
                    'batch_size': self.batch_size,
                    'window_size': self.window_size,
                    'padding': self.padding,
                    'imb_weight': self.imb_weight,
                    'weight_scale': self.weight_scale,
                    'patience': self.patience,
                    }, PATH)
        # self.best_state_dict_path = PATH+str(np.around(self.min_loss, decimals=5))+'_'+str(window_size)+'_'+str(self.lr_step_size)+'_'+str(self.batch_size)+'_'+str(self.dropout_p)+'_'+str(padding)+'.pth'

    def load_checkpoint(self,device, PATH='best', load_state_dict=True):
        if PATH == 'best':
            checkpoint = torch.load(self.save_model_path, map_location=device)
        else:
            checkpoint = torch.load(PATH, map_location=device)

        self.min_loss = checkpoint['loss_tracker']
        # print('model loss:', checkpoint['loss_tracker'])
        self.epoch = checkpoint['current_epoch']
        # print('epoch:', checkpoint['current_epoch'])
        self.dropout_p = checkpoint['dropout_p']
        # print('dropout_p:', checkpoint['dropout_p'])
        self.lr_step_size = checkpoint['lr_step_size']
        # print('lr_step_size:', checkpoint['lr_step_size'])
        self.out_features = checkpoint['out_features']
        # print('out_features:', checkpoint['out_features'])
        self.bidirectional = checkpoint['bidirectional']
        # print('bidirectional:', checkpoint['bidirectional'])
        self.val_loss_list = checkpoint['val_loss_list']
        # print('val_loss_list:', checkpoint['val_loss_list'])
        self.loss_list = checkpoint['loss_list']
        # print('loss_list:', checkpoint['loss_list'])
        self.batch_size = checkpoint['batch_size']
        # print('batch_size:', checkpoint['batch_size'])
        self.window_size = checkpoint['window_size']
        # print('window_size:', checkpoint['window_size'])
        self.padding = checkpoint['padding']
        # print('padding:', checkpoint['padding'])
        self.imb_weigth = checkpoint['imb_weight']
        # print('imb_weight:', checkpoint['imb_weight'])
        if load_state_dict: self.load_state_dict(checkpoint['net_state_dict'])
        else: return checkpoint['net_state_dict']

if __name__ == '__main__':
    import pandas as pd
    from tqdm import trange
    import pickle
    from pathlib import Path
    import numpy as np
    import os

    if torch.cuda.is_available():
        device = 'cuda'
    else: device = 'cpu'

    model_path = '/cluster/work/riner/patakiz/AML/task2/experiment_3/0.75592_0.60793_100_6000_128_0.2_0_0.6_0.9_6.pth'
    print(model_path)
    model = ExtendedNet(load_existing=True,
                        existing_path=model_path,
                        device=device).to(device)
    # # X = pd.read_hdf('x.h5', 'df')
    # # y = pd.read_hdf('y.h5', 'df').to_numpy()
    x_train = pd.read_csv("X_test_inv.csv", index_col=False)
    x_train_list = []
    # # count = 0
    for i in trange(x_train.shape[0]):
        x_train_list.append(x_train.loc[i].dropna().to_numpy(dtype='float32'))
    # print(len(x_test_list))
    #
    # X_list = []

    # print('Data preperation:')
    # for i in trange(y.shape[0]):
    #     X_list.append(X.loc[i].dropna().to_numpy(dtype='float32'))
    #
    # _, X_test_list, _, y_test = train_test_split(X_list, y, train_size=0.8, shuffle=False)
    # preds = []
    # model.eval()
    # with torch.no_grad():
    #     for i in trange(len(X_test_list)):
    #         pred = np.argmax(
    #             np.squeeze(model(torch.tensor(X_test_list[i][None, None, :]).to(device), True).cpu().numpy()))
    #         preds.append(pred)
    # preds = np.array(preds)

    #
    # X_test = pd.read_csv('X_test.csv', index_col='id')
    #
    #
    # # X_val = X_test.loc[found_id]
    #
    # x_test_list = []
    # # count = 0
    # # print(X_val.shape)
    # for i in found_id:
    #     x_test_list.append(X_test.loc[i].dropna().to_numpy(dtype='float32'))
    #
    #


    # preds = []
    # model.eval()
    # with torch.no_grad():
    #     for i in trange(len(X)):
    #         pred = np.argmax(
    #             np.squeeze(model(torch.tensor(X[i][None, None, :]).to(device), True).cpu().numpy()))
    #         preds.append(pred)
    # preds = np.array(preds)
    # test_accuracy = f1_score(y_test, preds, average='micro')
    # accuracy_mean = f1_score(y_test, preds, average='macro')
    # print(test_accuracy, accuracy_mean)

    out = []
    with torch.no_grad():
        for i in trange(len(x_train_list)):
            out.append(model(torch.tensor(x_train_list[i])[None, None, :].to(device), False).cpu().numpy())
    out = np.concatenate(out)
    print(out.shape)
    np.save("learned_features_test_inv.npy", out)


    # print(f1_score(y_test, preds, average='micro'))
    #
    # path = Path(model_path)
    # new_name = str(path.parent) + '/' + str(np.around(test_accuracy, decimals=5)) + '_' + str(
    #     np.around(accuracy_mean, decimals=5)) + '_' + str(path.name)
    #
    # os.rename(model_path, new_name)





