from torch import nn
import torch
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
from torch.optim import Adam

from sklearn.metrics import f1_score, accuracy_score, jaccard_score, confusion_matrix

import warnings
import math
import numpy as np
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from tqdm import tqdm
import biosppy.signals.ecg as ecg
from sklearn.model_selection import train_test_split

class MyDataset(Dataset):
    def __init__(self, length):
        self.length = length

    def __getitem__(self, idx):
        return idx

    def __len__(self):
        return self.length

class ExtendedNet(nn.Module):
    def __init__(self, lstm_features=None, lin_features=None, short_num_layers=None, long_num_layers=None, dropout=None, load_existing=False, existing_path=None, device=None):
        super(ExtendedNet, self).__init__()
        self.lstm_features = lstm_features
        self.lin_features = lin_features
        self.short_num_layers = short_num_layers
        self.long_num_layers = long_num_layers
        self.dropout_p = dropout

        self.conv1 = nn.Conv1d(in_channels=31, out_channels=self.lin_features, kernel_size=3, padding='same')
        # self.conv1 = nn.Conv1d(in_channels=31, out_channels=self.lin_features, kernel_size=, stride=2)
        self.conv2 = nn.Conv1d(in_channels=self.lin_features, out_channels=self.lstm_features, kernel_size=3, padding='same')
        # self.conv2 = nn.Conv1d(in_channels=self.lin_features, out_channels=self.lstm_features, kernel_size=3)

        self.lstm_short = nn.LSTM(100, self.lstm_features, num_layers=self.short_num_layers, dropout=self.dropout_p, batch_first=True)
        self.lstm_long_0 = nn.LSTM(self.lstm_features, self.lstm_features, num_layers=self.long_num_layers, dropout=self.dropout_p, batch_first=True)
        self.lstm_long_1 = nn.LSTM(self.lstm_features, self.lstm_features, num_layers=self.long_num_layers, dropout=self.dropout_p, batch_first=True)
        self.lstm_long_2 = nn.LSTM(self.lstm_features, self.lstm_features, num_layers=self.long_num_layers, dropout=self.dropout_p, batch_first=True)
        self.lstm_long_3 = nn.LSTM(self.lstm_features, self.lstm_features, num_layers=self.long_num_layers, dropout=self.dropout_p, batch_first=True)

        self.fc = nn.Linear(in_features=200, out_features=4)

        self.fc_1 = nn.Linear(in_features=self.lstm_features*3, out_features=200)
        self.fc_2 = nn.Linear(in_features=200, out_features=200)

        self.fc_test = nn.Linear(in_features=180, out_features=self.lstm_features)

        # self.t_fc1 = nn.Linear(in_features=31, out_features=self.lin_features)
        # self.t_fc2 = nn.Linear(in_features=self.lin_features, out_features=self.lstm_features)

        # self.t_fc_2 = nn.Linear(in_features=self.lstm_features*2, out_features=self.lin_features)
        # self.t_fc_3 = nn.Linear(in_features=self.lin_features, out_features=self.lstm_features)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(self.dropout_p)



        self.ll1 = nn.Linear(15, 100)
        self.ll2 = nn.Linear(100, 100)
        # self.ll3 = nn.Linear(300, 300)
        self.ll3 = nn.Linear(100, 1)
        self.ff1 = nn.Linear(31, 200)
        self.ff2 = nn.Linear(200, 200)
        self.ff3 = nn.Linear(200, 4)
        # if not load_existing:
        #     self.dropout_p = dropout
        #     self.out_features = out_features
        #     self.bidirectional = bidirectional
        # else:
        #     state_dict = self.load_checkpoint(device, existing_path, False)

        # self.dropout = nn.Dropout(p=self.dropout_p)

        # if load_existing: self.load_state_dict(state_dict)

    # def forward(self, X, X_expert, seq_len, classify, x_dummy=None):
    def forward(self, input, classify):
        # x, _ = self.lstm_short(X)
        # if not x_dummy is None:




        # X_dummy = self.relu(self.t_fc(X))
        # x_dummy, _ = self.lstm_short(X_dummy)
        #
        # x_dummy_2 = self.fc_test(X[:,:,0])
        # x_batches = []
        # for x in X_expert:
        #     x_batches.append())

        # x_batches = []
        # sum = 0
        # x_pad_packed = pad_packed_sequence(x)[0][-1, :, :]
        # for batch_len in seq_len:
        #     x_i = x_pad_packed[None, sum:sum + batch_len, :]
        #     sum += batch_len
        #     x_batches.append(x_i)

        # if not x_dummy is None:
        # x_batches = []
        # sum = 0
        # for batch_len in seq_len:
        #     x_i = x_dummy[sum:sum + batch_len, :][None,:, -1,:]
        #     x_i_2 = x_dummy_2[sum:sum + batch_len, :][None,:,:]
        #     sum += batch_len
        #     x_batches.append(torch.cat([x_i, x_i_2], dim=2))

        # x = self.dropout(self.relu(self.t_fc1(input.float().to(device))))
        # x = self.dropout(self.relu(self.t_fc2(x)))
        # print(input.shape)
        input = torch.permute(input.to(device).float(), (0, 2, 1))

        x = self.relu(self.ll1(input))
        x = self.relu(self.ll2(x))
        x = self.relu(self.ll3(x))
        # x = self.relu(self.ll4(x))
        x = self.relu(self.ff1(torch.squeeze(x)))
        x = self.relu(self.ff2(x))
        x = self.ff3(x)
        return x







        print(input.shape)
        x = self.dropout(self.relu(self.conv1(input.float().to(device))))
        # print(x.shape)
        x = self.dropout(self.relu(self.conv2(x)))
        x = torch.permute(x, (0, 2, 1))
        x, _ = self.lstm_long_0(x)
        y = torch.clone(x)
        z = torch.clone(x)
        x, _ = self.lstm_long_1(x)
        y, _ = self.lstm_long_2(y)
        z, _ = self.lstm_long_3(z)

        x = torch.mean(x, dim=1)
        y = y[:, -1, :]
        z = torch.max(z, dim=1)[0]
        # print(x.shape, y.shape, z.shape)

        # x_1 = []
        # x_2 = []
        # x_3 = []
        # # for x_batch in x_batches:
        # for x in X_expert:
        #     # print(x_batch.shape)
        #     # x_batch = self.dropout(self.relu(self.t_fc_2(x_batch)))
        #     # x_batch = self.dropout(self.relu(self.t_fc_3(x_batch)))
        #     x_batch = self.dropout(self.relu(self.t_fc1(x.float().to(device))))
        #     x_batch = self.dropout(self.relu(self.t_fc2(x_batch)))
        #     x_batch, _ = self.lstm_long_0(x_batch)
        #     x_i_1, _ = self.lstm_long_1(x_batch)
        #     x_1.append(torch.mean(x_i_1, dim=1))
        #     x_i_2, _ = self.lstm_long_2(x_batch)
        #     x_2.append(x_i_2[:, -1, :])
        #     x_i_3, _ = self.lstm_long_3(x_batch)
        #     x_3.append(torch.max(x_i_3, dim=1)[0])
        # x_1 = torch.cat(x_1)
        # x_2 = torch.cat(x_2)
        # x_3 = torch.cat(x_3)
        out = torch.cat([x,y,z], dim=1)
        # print(x.shape)
        out = self.dropout(self.relu(self.fc_1(out)))
        out = self.dropout(self.relu(self.fc_2(out)))
        if classify:
            out = self.fc(out)
        return out

    def extract_windows(self, X_, y_, window_size, padding):
        x_windows = []
        y_windows = []
        _, weight = np.unique(y_, return_counts=True)
        a = 2
        b = (1 + ((weight[0] / weight[1]) - 1) * self.imb_weight) * a
        c = (1 + ((weight[0] / weight[2]) - 1) * self.imb_weight) * a
        d = (1 + ((weight[0] / weight[3]) - 1) * self.imb_weight) * a
        ar = np.array([a,b,c,d])
        ar *= window_size/np.max(ar)
        a, b, c, d = int(ar[0]), int(ar[1]), int(ar[2]), int(ar[3])
        # for x in X_:
        #     print(x.shape)
        #     if x.shape[1]!=31:
        #         input(...)
        #
        # exit('DONE')
        # print(a,b,c,d)
        for x, y in zip(X_, y_):
            if x.shape[0] < window_size:
                if padding == '0':
                    padded = np.pad(x, ((window_size - x.shape[0] % window_size, 0),(0,0)))
                    x_windows.append(padded[None, :])

                    y_windows.append(y)
                elif padding == 'wrap':
                    padded = np.pad(x, ((window_size - x.shape[0] % window_size, 0),(0,0)), mode='wrap')
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
        weight_mid = (np.max(weight) - np.min(weight)) * 0.5 + np.min(weight)
        for i in range(weight.shape[0]):
            weight[i] = weight[i] + (weight_mid - weight[i]) * self.weight_scale
        print('damped class weights:', weight)
        weight = torch.tensor(weight).float()
        # weight[0] = 0.001

        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        return DataLoader(train_dataset, batch_size=batch_size), DataLoader(val_dataset, batch_size=batch_size), weight

    def train_net(self, X_list, X_list_expert, y, train_split, batch_size, n_epochs, lr_step_size, device, window_size, padding, imb_weight, weight_scale):
        self.train_split = train_split
        self.batch_size = batch_size
        self.lr_step_size = lr_step_size
        self.window_size = window_size
        self.weight_scale = weight_scale
        self.imb_weight = imb_weight
        self.padding = padding



        train_loader, val_loader, weight = self.get_dataloader(X_list_expert, y, train_split, self.window_size, self.batch_size, self.padding)


        # _, weight = np.unique(y, return_counts=True)
        # print(weight)
        # weight = 1 / torch.tensor(weight)
        # weight = weight / torch.sum(weight)
        # print(weight)
        optimizer = Adam(self.parameters(), lr=0.005)
        criterion = nn.CrossEntropyLoss(weight=weight).to(device)
        # criterion = nn.CrossEntropyLoss().to(device)
        # criterion = nn.CrossEntropyLoss().to(device)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=1/(2**0.5), patience=3, verbose=True)
        # scheduler = StepLR(optimizer, step_size=self.lr_step_size)

        self.accuracy_list = []
        iou_list = []
        self.loss_list = []
        self.max_accuracy = 0

        # X = []
        # for x in tqdm(X_list):
        #     X_i = []
        #     for template in x:
        #         X_i.append(torch.tensor(template[:, None]).float().to(device))
        #     X.append(X_i)
        # dataset = MyDataset(y.shape[0])
        # train_len = int(len(dataset)*train_split)
        # val_len = len(dataset)-train_len
        # train_dataset, val_dataset = random_split(dataset, [train_len, val_len])
        # train_loader = DataLoader(train_dataset, batch_size=batch_size)
        # val_loader = DataLoader(val_dataset, batch_size=batch_size)

        for self.epoch in range(n_epochs):
            running_loss = 0.0
            print('epoch:', self.epoch)

            self.train()
            for i, data in enumerate(tqdm(train_loader)):
                input, target = data
                # print(input.shape, target.shape)
                # X_packed = []
                # X_packed_expert = []
                # y_batch = []
                # seq_len = []
                # for j in indices.numpy():
                #     # X_packed+=X[j]
                #     for k in range(len(X[j])):
                #         X_packed.append(X[j][k][None, :, :])
                #     X_packed_expert.append(torch.tensor(X_list_expert[j])[None,:,:])
                #     y_batch.append(y[j])
                #     seq_len.append(len(X[j]))


                # X_packed = torch.cat(X_packed, dim=0)
                # X_packed_expert = torch.cat(X_packed_expert, dim=0)
                optimizer.zero_grad()
                # pred = self.forward(X_packed, X_packed_expert, seq_len, True)
                pred = self.forward(input, True)
                loss = criterion(pred.to(device), target.to(device))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()


            self.eval()
            with torch.no_grad():
                preds = []
                y_batch = []
                for i, data in enumerate(train_loader):
                    input, target = data
                    # X_packed = []
                    # X_packed_expert = []
                    # # X_packed_dummy = []
                    #
                    # seq_len = []
                    # for j in indices.numpy():
                    #     # X_packed += X[j]
                    #     for k in range(len(X[j])):
                    #         X_packed.append(X[j][k][None, :, :])
                    #     X_packed_expert.append(torch.tensor(X_list_expert[j])[None, :, :])
                    #     y_batch.append(y[j])
                    #     seq_len.append(len(X[j]))
                    #
                    #
                    # X_packed = torch.cat(X_packed, dim=0)
                    # X_packed = pack_sequence(X_packed)
                    # X_packed_dummy = torch.cat(X_packed_dummy, dim=0)

                    # pred = self.forward(X_packed, X_packed_expert, seq_len, True)
                    pred = self.forward(input, True)
                    preds.append(pred)
                    # print(pred.shape, target.shape)
                    y_batch.append(target)
                # y_batch = np.array(y_batch)

                preds = torch.cat(preds)
                y_batch = torch.cat(y_batch)
                # n0 = np.sum(preds==0)
                # n1 = np.sum(preds==1)
                # n2 = np.sum(preds==2)
                # n3 = np.sum(preds==3)
                # print(n0,n1,n2,n3)
                # n0 = np.sum(y_batch==0)
                # n1 = np.sum(y_batch==1)
                # n2 = np.sum(y_batch==2)
                # n3 = np.sum(y_batch==3)
                # print(n0,n1,n2,n3)
                # _, weight = np.unique(preds, return_counts=True)
                # print(weight)
                loss = criterion(preds.to(device), y_batch.to(device))

                accuracy = f1_score(y_batch, np.argmax(preds.cpu().numpy(), axis=1),  average='micro')
                accuracy_mean = f1_score(y_batch, np.argmax(preds.cpu().numpy(), axis=1), average='macro')
                confusion = confusion_matrix(y_batch, np.argmax(preds.cpu().numpy(), axis=1))

                # accuracy_ = jaccard_score(y_batch, preds,  average='micro')

            # pritn(f1_score(y_batch, preds, average='macro'))
            scheduler.step(loss.item())
            self.loss_list.append(running_loss)
            self.accuracy_list.append(loss.item())
            print(self.loss_list)
            print(self.accuracy_list)
            print('f1:', accuracy)
            print(confusion)
            # print(confusion[0,0]/np.sum(confusion[0,:]), confusion[1,1]/np.sum(confusion[1,:]),
            #       confusion[2,2]/np.sum(confusion[2,:]), confusion[3,3]/np.sum(confusion[3,:]))
            print('each f1:', confusion[0, 0] / (confusion[0, 0] + 0.5 * (
                        confusion[0, 1] + confusion[0, 2] + confusion[0, 3] + confusion[1, 0] + confusion[2, 0] +
                        confusion[3, 0])),
                  confusion[1, 1] / (confusion[1, 1] + 0.5 * (
                              confusion[1, 0] + confusion[1, 2] + confusion[1, 3] + confusion[0, 1] + confusion[2, 1] +
                              confusion[3, 1])),
                  confusion[2, 2] / (confusion[2, 2] + 0.5 * (
                              confusion[2, 0] + confusion[2, 1] + confusion[2, 3] + confusion[0, 2] + confusion[1, 2] +
                              confusion[3, 2])),
                  confusion[3, 3] / (confusion[3, 3] + 0.5 * (
                              confusion[3, 0] + confusion[3, 1] + confusion[3, 2] + confusion[0, 3] + confusion[1, 3] +
                              confusion[2, 3])))
            print('f1 mean:', accuracy_mean)

            # print(iou_list)




if __name__ == '__main__':
    import pandas as pd
    import pickle
    from tqdm import trange

    if torch.cuda.is_available(): device = 'cuda'
    else: device = 'cpu'
    print(device)
    model = ExtendedNet(lstm_features=128, short_num_layers=2, long_num_layers=1, lin_features=256, dropout=0.2).to(device)

    # X = pd.read_hdf('x.h5', 'df')
    y = pd.read_hdf('y.h5', 'df').to_numpy()
    #
    # X_list = []
    #
    # print('Data preperation:')
    # for i in trange(y.shape[0]):
    #     X_list.append(X.loc[i].dropna().to_numpy(dtype='float32'))
    #
    # templates_list = []
    # for i in trange(len(X_list)):
    #     _, _, _, _, templates, _, _ = ecg.ecg(=X_list[i], sampling_rate=300, show=False)
    #     templates_list.append(templates)
    #
    # with open('x_templates.pkl', 'wb') as handle:
    #     pickle.dump(templates_list, handle)

    with open('x_templates.pkl', 'rb') as handle:
        X_list = pickle.load(handle)
    #
    # X_packed = []
    # for i in trange(len(X_list)):
    #     pack_list = []
    #     for template in X_list[i]:
    #         pack_list.append(torch.tensor(template[:,None]).float())
    #     X_packed.append(pack_sequence(pack_list))

    # lstm = nn.LSTM(1, 10, batch_first=True)
    # a, _ = lstm(X_packed[-1])
    # b, _ = lstm(X_packed[-2])
    # print(pad_packed_sequence(a)[0].shape)
    with open("X_train_heartbeats.pkl", 'rb') as file:
        X_list_expert = pickle.load(file)

    X_train_list, X_test_list, X_train_list_expert, X_test_list_expert, y_train_list, y_test = train_test_split(X_list, X_list_expert, y, train_size=0.8, shuffle=False)

    model.train_net(X_train_list, X_train_list_expert, y_train_list, train_split=0.8, batch_size=64, n_epochs=200, lr_step_size=5, device=device, window_size=15, padding='skip_short', imb_weight=0.5, weight_scale=0.5)
