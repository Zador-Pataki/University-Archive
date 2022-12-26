import numpy as np
import torch
from ExtendedNet import ExtendedNet
import pandas as pd
from tqdm import trange
from sklearn.model_selection import train_test_split
import argparse
from sklearn.metrics import f1_score, confusion_matrix
from pathlib import Path
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--window_size", help='fusion strategy', type=int, required=True)
# parser.add_argument("--lr_step_size", help='skip_short', type=int, required=True)
parser.add_argument("--out_features", help='skip_short', type=int, required=True)
# parser.add_argument("--bidirectional", help='skip_short', type=str, required=True)
parser.add_argument("--padding", help='skip_short', type=str, required=True, choices=['wrap','skip_short', '0'])
parser.add_argument("--dropout", help='skip_short', type=float, required=True)
parser.add_argument("--imb_weight", help='skip_short', type=float, required=True)
parser.add_argument("--weight_scale", help='skip_short', type=float, required=True)
parser.add_argument("--patience", help='skip_short', type=int, required=True)
# parser.add_argument("--bias", help='skip_short', type=str, required=True)
args = parser.parse_args()

# if args.bidirectional == 'True': bidirectional = True
# else: bidirectional = False
# if args.bias == 'True': bias = True
# else: bias = False


if torch.cuda.is_available():
    # print('cuda devices:', torch.cuda.get_device_name(0), torch.cuda.get_device_name(1))
    device = 'cuda:0'
else:
    device = 'cpu'
print('device being used:', device)
experiment_path = '/cluster/work/riner/patakiz/AML/task2/experiment_3/'


X_train = pd.read_hdf('x.h5', 'df')
y_train = pd.read_hdf('y.h5', 'df').to_numpy()

X_train_list = []


print('Data preperation:')
for i in trange(y_train.shape[0]):
    X_train_list.append(X_train.loc[i].dropna().to_numpy(dtype='float32'))

# X_train_list, X_test_list, y_train, y_test = train_test_split(X_list, y, train_size=0.8, shuffle=False)

model = ExtendedNet(args.out_features, True, True, dropout=args.dropout).to(device)

model.train_net(X_train_list, y_train, train_split=0.85, n_epochs=150, experiment_path=experiment_path, device=device,
            lr_step_size=5, batch_size=128, window_size=args.window_size, padding=args.padding, imb_weight=args.imb_weight, weight_scale=args.weight_scale,
            patience=args.patience)

model.load_checkpoint(device=device, PATH='best')
preds = []



open_file = open('x_batman.pkl', "rb")
X = pickle.load(open_file)
open_file.close()
y_test = np.load('y_batman.npy')

preds = []
model.eval()
with torch.no_grad():
    for i in trange(len(X)):
        pred = np.argmax(
            np.squeeze(model(torch.tensor(X[i][None, None, :]).to(device), True).cpu().numpy()))
        preds.append(pred)
preds = np.array(preds)
test_accuracy = f1_score(y_test, preds, average='micro')
accuracy_mean = f1_score(y_test, preds, average='macro')
confusion = confusion_matrix(y_test, preds)

print('each f1:', confusion[0,0]/(confusion[0,0]+0.5*(confusion[0,1]+confusion[0,2]+confusion[0,3]+confusion[1,0]+confusion[2,0]+confusion[3,0])),
                  confusion[1, 1] / (confusion[1, 1] + 0.5 * (confusion[1, 0] + confusion[1, 2] + confusion[1, 3] + confusion[0,1] + confusion[2, 1] + confusion[3, 1])),
                  confusion[2, 2] / (confusion[2, 2] + 0.5 * (confusion[2, 0] + confusion[2, 1] + confusion[2, 3] + confusion[0,2] + confusion[1, 2] + confusion[3, 2])),
                  confusion[3, 3] / (confusion[3, 3] + 0.5 * (confusion[3, 0] + confusion[3, 1] + confusion[3, 2] + confusion[0,3] + confusion[1, 3] + confusion[2, 3])))
print('f1 mean:', accuracy_mean)
print('f1:', test_accuracy)

# model.eval()
# with torch.no_grad():
#     for i in trange(len(X_test_list)):
#         pred = np.argmax(np.squeeze(model(torch.tensor(X_test_list[i][None,None,:]).to(device), True).cpu().numpy()))
#         preds.append(pred)
# preds = np.array(preds)
#
# test_accuracy = f1_score(y_test, preds, average='micro')
# accuracy_mean = f1_score(y_test, preds, average='macro')
#
# confusion=confusion_matrix(y_test, preds)
# # print(test_accuracy)
# # print(confusion)
#
# print('each f1:', confusion[0,0]/(confusion[0,0]+0.5*(confusion[0,1]+confusion[0,2]+confusion[0,3]+confusion[1,0]+confusion[2,0]+confusion[3,0])),
#                   confusion[1, 1] / (confusion[1, 1] + 0.5 * (confusion[1, 0] + confusion[1, 2] + confusion[1, 3] + confusion[0,1] + confusion[2, 1] + confusion[3, 1])),
#                   confusion[2, 2] / (confusion[2, 2] + 0.5 * (confusion[2, 0] + confusion[2, 1] + confusion[2, 3] + confusion[0,2] + confusion[1, 2] + confusion[3, 2])),
#                   confusion[3, 3] / (confusion[3, 3] + 0.5 * (confusion[3, 0] + confusion[3, 1] + confusion[3, 2] + confusion[0,3] + confusion[1, 3] + confusion[2, 3])))
# print('f1 mean:', accuracy_mean)

path = Path(model.save_model_path)
new_name = str(path.parent)+'/'+str(np.around(test_accuracy, decimals=5))+'_'+str(np.around(accuracy_mean, decimals=5))+'_'+str(path.name)

os.rename(model.save_model_path, new_name)
