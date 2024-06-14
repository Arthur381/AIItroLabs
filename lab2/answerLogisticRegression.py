import numpy as np

# 超参数
# TODO: You can change the hyperparameters here
lr = 2  # 学习率
wd = 1e-2  # l2正则化项系数


def predict(X, weight, bias):
    """
    使用输入的weight和bias，预测样本X是否为数字0。
    @param X: (n, d) 每行是一个输入样本。n: 样本数量, d: 样本的维度
    @param weight: (d,)
    @param bias: (1,)
    @return: (n,) 线性模型的输出，即wx+b
    """
    # TODO: YOUR CODE HERE
    result=list()
    for x in X:
        result.append(np.dot(x,weight)+bias)
    var=np.array(result)
    ans=var.reshape(np.shape(var)[0],)
    #print("predict: ",ans.shape)
    print(np.shape(ans))
    return ans

    #raise NotImplementedError

def sigmoid(x):
    #print(x)
    if x<-(1e+2):
        return np.exp(x) / (np.exp(x) + 1)
    return 1 / (np.exp(-x) + 1)


def step(X, weight, bias, Y):
    """
    单步训练, 进行一次forward、backward和参数更新
    @param X: (n, d) 每行是一个训练样本。 n: 样本数量， d: 样本的维度
    @param weight: (d,)
    @param bias: (1,)
    @param Y: (n,) 样本的label, 1表示为数字0, -1表示不为数字0
    @return:
        haty: (n,) 模型的输出, 为正表示数字为0, 为负表示数字不为0
        loss: (1,) 由交叉熵损失函数计算得到
        weight: (d,) 更新后的weight参数
        bias: (1,) 更新后的bias参数
    """
    # TODO: YOUR CODE HERE
    predictArray=predict(X,weight,bias)
    num=np.shape(Y)[0]
    #print(np.shape(Y))
    fx=np.array(predictArray*Y)
    #print("fx elements: ",np.shape(fx[1]))
    loss=0
    for item in fx:
        if item<-(1e+3):
            loss += (((-item)+np.log(1+np.exp(item))) + wd * np.dot(weight, weight))
        else:
            loss += (-1) * (np.log(sigmoid(item)+(1e-6))) + wd * np.dot(weight, weight)

    loss=loss*(1/num)
    ppww=np.zeros(np.shape(X)[1])
    ppbb=0
    #print(np.shape(ppww))
    i=0
    for item in fx:
        #print("my shape: ",np.shape(((1-sigmoid(item))*Y[i])))
        #print("shape 1",np.shape(((1-sigmoid(item))*Y[i])*X[i]))
        if item<-(1e+3):
            tmp = ((1 -(np.exp(item)) / (np.exp(item) + 1) ) * Y[i])
        else:
            #if (item < -(1e+6)):
            #    print("wrong")
            tmp=((1-sigmoid(item))*Y[i])
        ppww+=tmp*X[i]
        ppbb+=tmp
        i+=1
    Pweight=-(1/num)*ppww+2*wd*weight
    #Pweight=-np.mean(((1-np.exp(fx))/(np.exp(fx)+1))*Y*X.T,axis=0)
    Pbias=-(1/num)*ppbb
    Newweight,Newbias=weight-lr*Pweight,bias-lr*Pbias
    return  (np.array(predictArray),np.array(loss),np.array(Newweight),np.array(Newbias))
    #raise NotImplementedError
