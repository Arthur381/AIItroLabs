import numpy as np
from copy import deepcopy
from typing import List, Callable

EPS = 1e-6

# 超参数，分别为树的最大深度、熵的阈值、信息增益函数
# TODO: You can change or add the hyperparameters here
hyperparams = {
    "depth": 6,
    "purity_bound": 1,
    "gainfunc": "negginiDA"
    }

def entropy(Y: np.ndarray):
    """
    计算熵
    @param Y: (n,), 标签向量
    @return: 熵
    """
    # TODO: YOUR CODE HERE
    ufeat, featcnt=np.unique(Y,return_counts=True)
    featp=featcnt/Y.shape[0]#占比向量
    ans=0
    for index in range(len(ufeat)):
        ans+=(featp[index])*np.log2(featp[index])
    return -ans
    #没有问题
    #raise NotImplementedError


def gain(X: np.ndarray, Y: np.ndarray, idx: int):#计算某一个idx对应的信息增益
    """
    计算信息增益
    @param X: (n, d), 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: (n,), 样本的label
    @param idx: 第idx个特征
    @return: 信息增益
    """
    feat = X[:, idx]#取出x的某一列
    ufeat, featcnt = np.unique(feat, return_counts=True)
    #print("ufeat",np.shape(ufeat),ufeat)
    #print("featcnt",np.shape(featcnt),featcnt)
    featp = featcnt / feat.shape[0]
    ret = 0
    # TODO: YOUR CODE HERE
    index=0
    if  Y.shape[0]!=X.shape[0]:
        print("wrong")
    #print(feat.shape[0])
    for item in ufeat:
        part=[Y[apart] for apart in range(feat.shape[0]) if feat[apart]==item]#提取出来
        part=np.array(part)#part 是y中的一部分
        ret+=(featp[index])*entropy(part+EPS)#比例*部分信息熵
    #raise NotImplementedError
    ret=entropy(Y+EPS)-ret
        #为什么增益这么小
    return ret


def gainratio(X: np.ndarray, Y: np.ndarray, idx: int):
    """
    计算信息增益比
    @param X: (n, d), 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: (n,), 样本的label
    @param idx: 第idx个特征
    @return: 信息增益比
    """
    ret = gain(X, Y, idx) / (entropy(X[:, idx]) + EPS)
    return ret


def giniD(Y: np.ndarray):
    """
    计算基尼指数
    @param Y: (n,), 样本的label
    @return: 基尼指数
    """
    u, cnt = np.unique(Y, return_counts=True)
    p = cnt / Y.shape[0]#该指标中不同类 的占比向量
    return 1 - np.sum(np.multiply(p, p))


def negginiDA(X: np.ndarray, Y: np.ndarray, idx: int):
    """
    计算负的基尼指数增益
    @param X: (n, d), 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: (n,), 样本的label
    @param idx: 第idx个特征
    @return: 负的基尼指数增益
    """
    feat = X[:, idx]
    ufeat, featcnt = np.unique(feat, return_counts=True)
    featp = featcnt / feat.shape[0]
    ret = 0
    for i, u in enumerate(ufeat):
        mask = (feat == u)
        ret -= featp[i] * giniD(Y[mask])
    ret += giniD(Y)  # 调整为正值，便于比较
    return ret


class Node:
    """
    决策树中使用的节点类
    """
    def __init__(self): 
        self.children = {}          # 子节点列表，其中key是特征的取值，value是子节点（Node）
        self.featidx: int = None    # 用于划分的特征
        self.label: int = None      # 叶节点的标签

    def isLeaf(self):
        """
        判断是否为叶节点
        @return: bool, 是否为叶节点
        """
        return len(self.children) == 0#没有孩子节点
#没有取到划分特征？

def buildTree(X: np.ndarray, Y: np.ndarray, unused: List[int], depth: int, purity_bound: float, gainfunc: Callable, prefixstr=""):
    """
    递归构建决策树。
    @params X: (n, d), 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @params Y: (n,), 样本的label
    @params unused: List of int, 未使用的特征索引
    @params depth: int, 树的当前深度
    @params purity_bound: float, 熵的阈值
    @params gainfunc: Callable, 信息增益函数
    @params prefixstr: str, 用于打印决策树结构
    @return: Node, 决策树的根节点
    """
    
    root = Node()
    u, ucnt = np.unique(Y, return_counts=True)
    root.label = u[np.argmax(ucnt)]#取出现次数最多的指标,到达最深层，叶子节点可能标签不唯一
    print(prefixstr, f"label {root.label} numbers {u} count {ucnt}") #可用于debug
    # 当达到终止条件时，返回叶节点
    # TODO: YOUR CODE HERE
    if depth==0 or entropy(Y)<purity_bound:#达到某个深度或者足够纯净
        return root
    #raise NotImplementedError
    gains = [gainfunc(X, Y, i) for i in unused]#gain 是可设置的，根据unsued 加入列表
    idx = np.argmax(gains)#在取了呀？
    root.featidx = unused[idx]
    unused = deepcopy(unused)
    unused.pop(idx)
    feat = X[:, root.featidx]#选择某一个最适指标
    ufeat = np.unique(feat)#这个指标下，划分为len（ufeat）个叶子节点
    # 按选择的属性划分样本集，递归构建决策树
    # 提示：可以使用prefixstr来打印决策树的结构
    # TODO: YOUR CODE HERE
    #if len(ufeat)>1:
    #    print("clean")
    for leaf in ufeat:
        sonY=np.array([Y[i] for i in range(Y.shape[0]) if feat[i]==leaf])
        #print(Y.shape)
        #print(entropy(Y))
        sonX=np.array([X[i] for i in range(Y.shape[0]) if feat[i]==leaf])
        root.children[leaf]=buildTree(sonX,sonY,unused,depth-1,purity_bound,gainfunc,prefixstr=' ')
    #raise NotImplementedError
    #print(root.children)
    return root


def inferTree(root: Node, x: np.ndarray):
    """
    利用建好的决策树预测输入样本为哪个数字
    @param root: 当前推理节点
    @param x: d*1 单个输入样本
    @return: int 输入样本的预测值
    """
    if root.isLeaf():
        return root.label#
    child = root.children.get(x[root.featidx], None)
    return root.label if child is None else inferTree(child, x)

