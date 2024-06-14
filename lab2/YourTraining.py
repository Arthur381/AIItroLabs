'''
Softmax 回归。计算accuracy。
'''
import mnist
import numpy as np
import scipy.ndimage as ndimage
import pickle
from autograd.utils import PermIterator
from util import setseed
import mnist
from copy import deepcopy
from typing import List
from autograd.BaseGraph import Graph
from autograd.utils import buildgraph
from autograd.BaseNode import *


'''trn_X = np.load('MNIST/train_data.npy').astype(np.float64)
trn_Y = np.load('MNIST/train_targets.npy')
#print(trn_X.shape)
trn_num_sample = trn_X.shape[0]
trn_X = trn_X.reshape(trn_num_sample, -1)
#print(trn_X.shape)这里使他们变成二维的
std_X, mean_X = np.std(trn_X, axis=0, keepdims=True)+1e-4, np.mean(trn_X, axis=0, keepdims=True)
num_feat = trn_X.shape[1]
num_class = np.max(trn_Y) + 1'''

setseed(0)

lr = 0.001   # 学习率
wd1 = 1e-5  # L1正则化
wd2 = 1e-5  # L2正则化
batchsize = 128
#训练集和测试集合并

val_X = np.load('MNIST/valid_data.npy').astype(np.float64)
val_Y = np.load('MNIST/valid_targets.npy')

val_num_sample = val_X.shape[0]

X = val_X.reshape(val_X.shape[0], -1)
Y = val_Y.reshape(val_Y.shape[0], -1)

num_feat = X.shape[1]
num_class = np.max(Y) + 1
X = val_X[:]
Y = val_Y[:]

shifts = np.random.randint(-4, 5, size=(val_num_sample, 2))
angles = np.random.uniform(-30, 30, size=val_num_sample)
#print(angles.shape, shifts.shape)
for id in range(val_num_sample):
    ndimage.shift(X[id], shifts[id], output=X[id])
    cur_X = ndimage.rotate(X[id], angles[id], reshape=False)
    X[id] = cur_X

# 这里才展平
X = X.reshape(X.shape[0], -1)
Y = Y.reshape(Y.shape[0])
def buildGraph(Y):
    """
    建图
    @param Y: n 样本的labelat
    @return: Graph类的实例, 建好的图
    """
    nodes = [BatchNorm(784),
             Linear(num_feat,512), relu(),Dropout_Corrected(0.28),
             Linear(512, 128), relu(), Dropout_Corrected(0.28),
             #Linear(128, 64), relu(), Dropout_Corrected(0.28), #, BatchNorm(64),
             Linear(128,num_class), Softmax(),
             CrossEntropyLoss(Y)]
    graph = Graph(nodes)
    return graph


#setseed(0) # 固定随机数种子以提高可复现性

save_path = "model/mtr.npy"

if __name__ == "__main__":
    graph = buildGraph(Y)
    # 训练
    best_train_acc = 0
    dataloader = PermIterator(X.shape[0], batchsize)
    for i in range(1, 40+1):# epoch 的数量
        hatys = []
        ys = []
        losss = []
        #print(X.shape, Y.shape)
        graph.train()
        for perm in dataloader:
            tX = X[perm]
            tY = Y[perm]
            graph[-1].y = tY
            graph.flush()
            pred, loss = graph.forward(tX)[-2:]
            hatys.append(np.argmax(pred, axis=1))
            ys.append(tY)
            graph.backward()
            graph.optimstep(lr, wd1, wd2)
            losss.append(loss)
        loss = np.average(losss)
        acc = np.average(np.concatenate(hatys)==np.concatenate(ys))
        print(f"epoch {i} loss {loss:.3e} acc {acc:.4f}")
        if acc > best_train_acc:
            best_train_acc = acc
            with open(save_path, "wb") as f:
                pickle.dump(graph, f)#将graph存入

        '''test_X = np.load('MNIST/test_data.npy').astype(np.float64)
        test_Y = np.load('MNIST/test_targets.npy')

        num_data = test_X.shape[0]

        with open(save_path, "rb") as f:
            graph = pickle.load(f)
        graph.eval()
        graph.flush()
        test_X = test_X.reshape(test_X.shape[0], -1)  # 展平
        print(test_X.shape)
        print(test_Y.shape)
        pred = graph.forward(test_X, removelossnode=1)[-1]
        haty = np.argmax(pred, axis=1)
        print("valid acc", np.average(haty == test_Y))'''
