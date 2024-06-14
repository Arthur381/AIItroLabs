from typing import List
import math
import numpy as np
import numpy as np
from .Init import * 

def shape(X):
    if isinstance(X, np.ndarray):
        ret = "ndarray"
        if np.any(np.isposinf(X)):
            ret += "_posinf"
        if np.any(np.isneginf(X)):
            ret += "_neginf"
        if np.any(np.isnan(X)):
            ret += "_nan"
        return f" {X.shape} "
    if isinstance(X, int):
        return "int"
    if isinstance(X, float):
        ret = "float"
        if np.any(np.isposinf(X)):
            ret += "_posinf"
        if np.any(np.isneginf(X)):
            ret += "_neginf"
        if np.any(np.isnan(X)):
            ret += "_nan"
        return ret
    else:
        raise NotImplementedError(f"unsupported type {type(X)}")

class Node(object):#这里使用了类继承
    def __init__(self, name, *params):
        self.grad = [] # 节点的梯度，self.grad[i]对应self.params[i]在反向传播时的梯度
        self.cache = [] # 节点保存的临时数据
        self.name = name # 节点的名字
        self.params = list(params) # 用于Linear节点中存储 weight和 bias参数使用

    def num_params(self):
        return len(self.params)

    def cal(self, X):
        '''
        计算函数值。请在其子类中完成具体实现。
        '''
        pass

    def backcal(self, grad):
        '''
        计算梯度。请在其子类中完成具体实现。
        '''
        pass

    def flush(self):
        '''
        初始化或刷新节点内部数据，包括梯度和缓存
        '''
        self.grad = []
        self.cache = []

    def forward(self, X, debug=False):
        '''
        正向传播。输入X，输出正向传播的计算结果。
        '''
        if debug:
            print(self.name, shape(X))
        ret = self.cal(X)
        if debug:
            print(shape(ret))
        return ret

    def backward(self, grad, debug=False):
        '''
        反向传播。输入grad（该grad为反向传播到该节点的梯度），输出反向传播到下一层的梯度。
        '''
        if debug:
            print(self.name, shape(grad))
        ret = self.backcal(grad)
        if debug:
            print(shape(ret))
        return ret
    
    def eval(self):
        pass

    def train(self):
        pass


class relu(Node):
    # input X: (*)，即可能是任意维度
    # output relu(X): (*)
    def __init__(self):
        super().__init__("relu")

    def cal(self, X):
        self.cache.append(X)
        return np.clip(X, 0, None)

    def backcal(self, grad):
        return np.multiply(grad, self.cache[-1] > 0) 



class sigmoid(Node):
    # input X: (*)，即可能是任意维度
    # output sigmoid(X): (*)#返回值为相同维度
    def __init__(self):
        super().__init__("sigmoid")#这里初始化名字

    #def sigmoidF(self,X):

    def cal(self, X):
        # TODO: YOUR CODE HERE
        expX=np.exp(X)
        ret=expX/(expX+1)
        self.cache.append(ret)
        return ret

        #raise NotImplementedError

    def backcal(self, grad):
        # TODO: YOUR CODE HERE
        return np.multiply(grad,np.multiply(self.cache[-1], 1-self.cache[-1]))

        #raise NotImplementedError
    
class tanh(Node):
    # input X: (*)，即可能是任意维度
    # output tanh(X): (*)
    def __init__(self):
        super().__init__("tanh")

    def cal(self, X):
        ret = np.tanh(X)
        self.cache.append(ret)
        return ret


    def backcal(self, grad):#矩阵对应元素位置相乘
        return np.multiply(grad, np.multiply(1+self.cache[-1], 1-self.cache[-1]))#该函数的导数
    

class Linear(Node):#线性
    # input X: (*,d1)
    # param weight: (d1, d2)
    # param bias: (d2)
    # output Linear(X): (*, d2)
    def __init__(self, indim, outdim):
        """
        初始化
        @param indim: 输入维度
        @param outdim: 输出维度
        """
        weight = kaiming_uniform(indim, outdim)
        bias = zeros(outdim)
        super().__init__("linear", weight, bias)#超类实例化

    def cal(self, X):
        # TODO: YOUR CODE HERE
        ret=np.dot(X,self.params[0])+self.params[1]
        self.cache.append(X)
        return ret
        #raise NotImplementedError

    def backcal(self, grad):
        '''
        需要保存 weight 和 bias 的梯度，可以参考Node类和BatchNorm类
        '''
        # TODO: YOUR CODE HERE
        # w有 d2 个列，代表该层有 d2 个神经元
        #此时注意：我已经储存了梯度，但是没有改变 weight 和 bias
        #不知道标签，如何计算 weight 和 bias 的梯度
        gradW = np.dot(self.cache[-1].T,grad)#
        gradB = np.sum(grad,axis=0)
        #print(np.shape(gradW),"gradW")
        self.grad.append(gradW)
        self.grad.append(gradB)
        return np.dot(grad,self.params[0].T)#
        #raise NotImplementedError


class StdScaler(Node):
    '''
    input shape (*)
    output (*)
    '''
    EPS = 1e-3
    def __init__(self, mean, std):
        super().__init__("StdScaler")
        self.mean = mean
        self.std = std

    def cal(self, X):
        X = X.copy()
        X -= self.mean
        X /= (self.std + self.EPS)
        return X

    def backcal(self, grad):
        return grad/ (self.std + self.EPS)
    


class BatchNorm(Node):#批归一化
    '''
    input shape (*)
    output (*)
    '''
    EPS = 1e-8
    def __init__(self, indim, momentum: float = 0.9):
        super().__init__("batchnorm", ones((indim)), zeros(indim))
        self.momentum = momentum
        self.mean = None
        self.std = None
        self.updatemean = True
        self.indim = indim

    def cal(self, X):
        if self.updatemean:#求每一列的均值和标准差
            tmean, tstd = np.mean(X, axis=0, keepdims=True), np.std(X, axis=0, keepdims=True)
            if self.mean is None or self.std is None:
                self.mean = tmean
                self.std = tstd
            else:
                self.mean *= self.momentum
                self.mean += (1-self.momentum) * tmean
                self.std *= self.momentum
                self.std += (1-self.momentum) * tstd
        X = X.copy()
        X -= self.mean
        X /= (self.std + self.EPS)
        self.cache.append(X.copy())
        X *= self.params[0]
        X += self.params[1]
        return X

    def backcal(self, grad):
        X = self.cache[-1]
        self.grad.append(np.multiply(X, grad).reshape(-1, self.indim).sum(axis=0))
        self.grad.append(grad.reshape(-1, self.indim).sum(axis=0))
        return (grad*self.params[0])/ (self.std + self.EPS)
    
    def eval(self):
        self.updatemean = False

    def train(self):
        self.updatemean = True


class Dropout_Corrected(Node):
    '''
    input shape (*)
    output (*)
    '''

    def __init__(self, p: float = 0.1):
        super().__init__("dropout")
        assert 0 <= p <= 1, "p 是dropout 概率，必须在[0, 1]中"
        self.p = p
        self.dropout = True

    def cal(self, X):
        if self.dropout:
            X = X.copy()
            mask = np.random.rand(*X.shape) < self.p
            np.putmask(X, mask, 0)
            X = X * (1 / (1 - self.p))
            self.cache.append(mask)
        return X

    def backcal(self, grad):
        if self.dropout:
            grad = grad.copy()
            np.putmask(grad, self.cache[-1], 0)
            grad = grad * (1 / (1 - self.p))
        return grad

    def eval(self):
        self.dropout = False

    def train(self):
        self.dropout = True


class Dropout(Node):#只激活一部分神经元
    '''
    input shape (*)
    output (*)
    '''
    def __init__(self, p: float = 0.1):
        super().__init__("dropout")
        assert 0<=p<=1, "p 是dropout 概率，必须在[0, 1]中"
        self.p = p
        self.dropout = True

    def cal(self, X):
        if self.dropout:
            X = X.copy()
            mask = np.random.rand(*X.shape) < self.p
            np.putmask(X, mask, 0)
            self.cache.append(mask)
        else:
            X = X*(1/(1-self.p))
        return X
    
    def backcal(self, grad):
        if self.dropout:
            grad = grad.copy()
            np.putmask(grad, self.cache[-1], 0)
            return grad
        else:
            return (1/(1-self.p)) * grad
    
    def eval(self):
        self.dropout=False

    def train(self):
        self.dropout=True

class Softmax(Node):
    # input X: (*) #X为任意维度
    # output softmax(X): (*), softmax at 'dim'
    def __init__(self, dim=-1):
        super().__init__("softmax")
        self.dim = dim

    def cal(self, X):#keepdim 保持原来的空间特性？？？？
        X = X - np.max(X, axis=self.dim, keepdims=True)#我们这里找到最大的
        expX = np.exp(X)
        ret = expX / expX.sum(axis=self.dim, keepdims=True)
        self.cache.append(ret)
        return ret

    def backcal(self, grad):
        softmaxX = self.cache[-1]
        grad_p = np.multiply(grad, softmaxX)
        return grad_p - np.multiply(grad_p.sum(axis=self.dim, keepdims=True), softmaxX)


class LogSoftmax(Node):#对数softmax
    # input X: (*)
    # output logsoftmax(X): (*), logsoftmax at 'dim'


    def __init__(self, dim=-1):
        super().__init__("logsoftmax")
        self.dim = dim

    def cal(self, X):
        # TODO: YOUR CODE HERE
        EPS = 1e-8
        X = X - np.max(X, axis=self.dim, keepdims=True)  # 我们这里找到最大的去除之后，防止爆炸，但是结果不变
        expX = np.exp(X)#错因：取了以二为底的对数，可是又使用了 exp
        ret = np.log(expX / expX.sum(axis=self.dim, keepdims=True)+EPS)
        self.cache.append(ret)
        return ret

        #raise NotImplementedError

    def backcal(self, grad):
        # TODO: YOUR CODE HERE
        logsoftmaxX = self.cache[-1]
        return grad - np.multiply(grad.sum(axis=self.dim, keepdims=True), np.exp(logsoftmaxX))
        #raise NotImplementedError




class NLLLoss(Node):#负对数似然损失
    '''
    negative log-likelihood 损失函数
    '''
    # shape X: (*, d), y: (*)
    # shape value: number 
    # 输入：X: (*) 个预测，每个预测是个d维向量，代表d个类别上分别的log概率。  y：(*) 个整数类别标签
    # 输出：NLL损失
    def __init__(self, y):
        """
        初始化
        @param y: n 样本的label
        """
        super().__init__("NLLLoss")
        self.y = y

    def cal(self, X):
        y = self.y
        self.cache.append(X)
        return - np.sum(np.take_along_axis(X, np.expand_dims(y, axis=-1), axis=-1))#找到对应的x，最终只求分队log和

    def backcal(self, grad):
        X, y = self.cache[-1], self.y
        ret = np.zeros_like(X)#最初全部设置为零
        np.put_along_axis(ret, np.expand_dims(y, axis=-1), -1, axis=-1)
        return grad * ret
    #np.put_along_axis (取出矩阵的一部分)
    #np.expand_dims (作用在最小单位)


class CrossEntropyLoss(Node):#交叉熵损失
    '''
    多分类交叉熵损失函数，不同于课上讲的二分类。它与NLLLoss的区别仅在于后者输入log概率，前者输入概率。
    '''
    # Loss=−∑ni=1 (yi* log yi^)
    # shape X: (*, d), y: (*)
    # shape value: number 
    # 输入：X: (*) 个预测，每个预测是个d维向量，代表d个类别上分别的概率。  y：(*) 个整数类别标签
    # 输出：交叉熵损失
    def __init__(self, y):
        """
        初始化
        @param y: n 样本的label
        """
        super().__init__("CELoss")
        self.y = y

    def cal(self, X):
        # TODO: YOUR CODE HERE
        EPS=1e-6
        # 提示，可以对照NLLLoss的cal
        y = self.y
        logX=np.log(X+EPS)#先取对数
        self.cache.append(X)
        return - np.sum(np.take_along_axis(logX, np.expand_dims(y, axis=-1), axis=-1))
        #raise NotImplementedError

    def backcal(self, grad):
        # TODO: YOUR CODE HERE
        EPS = 1e-6
        # 提示，可以对照NLLLoss的backcal
        X, y = self.cache[-1], self.y
        ret = np.zeros_like(X)
        np.put_along_axis(ret, np.expand_dims(y, axis=-1), -1, axis=-1)
        return (grad * ret)/(X+EPS)


        #raise NotImplementedError