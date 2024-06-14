import math
from SST_2.dataset import traindataset, minitraindataset
from fruit import get_document, tokenize
import pickle
import numpy as np
from importlib.machinery import SourcelessFileLoader
from autograd.BaseGraph import Graph
from autograd.BaseNode import *

np.random.seed(1)

class NullModel:
    def __init__(self):
        pass

    def __call__(self, text):
        return 0


class NaiveBayesModel:#不考虑上下文关系
    def __init__(self):
        #标签和语句分别以怎样的形式表出
        self.dataset = traindataset(shuffle=False) # 完整训练集，需较长加载时间
        #self.dataset = minitraindataset(shuffle=False) # 用来调试的小训练集，仅用于检查代码语法正确性
        # 以下内容可根据需要自行修改，不修改也可以完成本题
        self.token_num = [{}, {}] # token在正负样本中出现次数
        #列表中包含了两个字典
        self.V = 0 #语料库token数量,
        #可能用于 Laplace 平滑
        self.pos_neg_num = [0, 0] # 正负样本数量
        self.count()#已经将训练集中可以统计到的东西加入


    def count(self):#标签 1 为正样本，index 为[0]
        # TODO: YOUR CODE HERE
        # 提示：统计token分布不需要返回值
        l=self.dataset.len#得到训练集的长度
        #t=0
        for toks,lab in self.dataset:
            have_search=set()
            if lab==1:#为正样本
                for word in toks:#一句话中如果有多个单词，则不重复统计
                    if word not in have_search:
                        have_search.add(word)
                        if (word not in self.token_num[0]) and (word not in self.token_num[1]):#新的 token
                            self.V+=1
                        if word in self.token_num[0]:#如果在字典里
                            self.token_num[0][word]+=1
                        else:
                            self.token_num[0][word]=1
                self.pos_neg_num[0]+=1
            elif lab==0:#为负样本
                for word in toks :
                    if word not in have_search:
                        have_search.add(word)
                        if (word not in self.token_num[0]) and (word not in self.token_num[1]):
                            self.V+=1
                        if word in self.token_num[1]:#如果在字典里
                            self.token_num[1][word]+=1
                        else:
                            self.token_num[1][word]=1
                self.pos_neg_num[1]+=1
        #raise NotImplementedError # 填写完成后删除这句

    def __call__(self, text):
        # TODO: YOUR CODE HERE
        # 返回1或0代表当前句子分类为正/负样本
        a=1#lapace 平滑参数

        #print("token_ num",self.token_num)
        Sum=self.pos_neg_num[0]+self.pos_neg_num[1]#样本数
        minpro=1/(10*Sum)
        p_good=self.pos_neg_num[0]/Sum
        p_bad=self.pos_neg_num[1]/Sum
        #print("text:",text)
        cleaned_toks=text
        for a_token in cleaned_toks :
            #print("Good:",p_good)
            #print("Bad:",p_bad)
            p_good*=(self.token_num[0].get(a_token, minpro)+a)/(self.pos_neg_num[0]+self.V*a)
            p_bad*=(self.token_num[1].get(a_token, minpro)+a)/(self.pos_neg_num[1]+self.V*a)
            #print("in good",self.token_num[0].get(a_token, minpro))
            #print("in bad",self.token_num[1].get(a_token, minpro))
            #if(a_token not in self.token_num[0] and a_token not in self.token_num[1]):
                #print("exc",minpro)
            
        
        if p_good>p_bad:
            return 1
        else:
            return 0
        #raise NotImplementedError


def buildGraph(dim, num_classes, L): #dim: 输入一维向量长度, num_classes:分类数
    # 以下类均需要在BaseNode.py中实现
    # 也可自行修改模型结构
    nodes = [Attention(dim), relu(), LayerNorm((L, dim)), ResLinear(dim), relu(), LayerNorm((L, dim)), Mean(1), Linear(dim, num_classes), LogSoftmax(), NLLLoss(num_classes)]
    #nodes = [Attention(dim), relu(), LayerNorm((L, dim)),
    #         ResLinear(dim), relu(),Mean(1),
    #        Linear(dim, num_classes), LogSoftmax(), NLLLoss(num_classes)]
    
    graph = Graph(nodes)
    return graph


save_path = "model/attention.npy"

class Embedding():
    def __init__(self):
        self.emb = dict() 
        with open("words.txt", encoding='utf-8') as f: #word.txt存储了每个token对应的feature向量，self.emb是一个存储了token-feature键值对的Dict()，可直接调用使用
            for i in range(50000):
                row = next(f).split()
                word = row[0]
                vector = np.array([float(x) for x in row[1:]])
                self.emb[word] = vector
        
    def __call__(self, text, max_len=50):
        # TODO: YOUR CODE HERE
        # 利用self.emb将句子映射为一个二维向量（LxD），注意，同时需要修改训练代码中的网络维度部分
        # 默认长度L为50，特征维度D为100
        ##可能是单词的个数为 50
        # 提示: 考虑句子如何对齐长度，且可能存在空句子情况（即所有单词均不在emd表内） 
        ans=np.zeros((max_len,100))
        # self.emb 长什么样子？
        #zer=list(zer)
        index=0
        for word in text:
            if word in self.emb:
                ans[index]=self.emb[word]
            index+=1
            if index==max_len:
                break
        return ans
        #raise NotImplementedError


class AttentionModel():
    def __init__(self):
        self.embedding = Embedding()
        with open(save_path, "rb") as f:
            self.network = pickle.load(f)
        self.network.eval()
        self.network.flush()

    def __call__(self, text, max_len=50):
        X = self.embedding(text, max_len)
        X = np.expand_dims(X, 0)
        pred = self.network.forward(X, removelossnode=1)[-1]
        haty = np.argmax(pred, axis=-1)
        return haty[0]


class QAModel():
    def __init__(self):
        self.document_list = get_document()

    def tf(self, word, document):
        # TODO: YOUR CODE HERE
        # 返回单词在文档中的频度
        # document变量结构请参考fruit.py中get_document()函数
        cur_doc=self.document_list[document]
        cleaned_tokens=cur_doc['document']
        N=len(cleaned_tokens)
        n=0
        for i in range(N):
            if cleaned_tokens[i]==word:
                n+=1
        return np.log10(n/N+1)

        #raise NotImplementedError  

    def idf(self, word):#总共在多少个文件中出现
        # TODO: YOUR CODE HERE
        # 返回单词IDF值，提示：你需要利用self.document_list来遍历所有文档
        # 注意python整除与整数除法的区别
        D=len(self.document_list)
        d=0
        for a_doc in self.document_list:
            if word in a_doc['document']:
                d+=1
        return np.log10(D/(d+1))


        #raise NotImplementedError
    
    def tfidf(self, word, document):
        # TODO: YOUR CODE HERE
        # 返回TF-IDF值
        return self.idf(word)*self.tf(word,document)
        #raise NotImplementedError

    def __call__(self, query):
        query = tokenize(query) # 将问题token化
        # TODO: YOUR CODE HERE
        # 利用上述函数来实现QA
        # 提示：你需要根据TF-IDF值来选择一个最合适的文档，再根据IDF值选择最合适的句子
        # 返回时请返回原本句子，而不是token化后的句子，数据结构请参考README中数据结构部分以及fruit.py中用于数据处理的get_document()函数
        choosedoc=dict()
        i=0#在 List 中遍历，与指标一一对应
        l1=len(self.document_list)
        for i in range(l1):#统计单词在文档中比较最大的 tf-idf
            doc_vec = list()
            for a_word in query:
                val=self.tfidf(a_word,i)
                doc_vec.append(val)
            choosedoc[i]=sum(doc_vec)#计算和
            i += 1
        real_doc=self.document_list[max(choosedoc,key=choosedoc.get)]
        #选择句子
        choosesen=dict()
        sentences=real_doc['sentences']#类型是 List
        l2=len(sentences)
        for i in range(l2):#选择一个句子
            sum_idf=0
            num = 0
            for a_word in query:
                if a_word in sentences[i][0]:
                    sum_idf+=self.idf(a_word)#同时在句子里
                    num+=1#共有的单词数
            choosesen[i]=sum_idf+sum_idf*(num/len(query))
            i+=1
        real_sen = sentences[max(choosesen, key=choosesen.get)]
        return real_sen[1]
        #raise NotImplementedError


modeldict = {
    "Null": NullModel,
    "Naive": NaiveBayesModel,
    "Attn": AttentionModel,
    "QA": QAModel,
}


if __name__ == '__main__':
    embedding = Embedding()
    lr = 5e-3   # 学习率
    wd1 = 1e-4  # L1正则化
    wd2 = 1e-5  # L2正则化
    batchsize = 64
    max_epoch = 15
    
    max_L = 50
    num_classes = 2
    feature_D = 100

    graph = buildGraph(feature_D, num_classes, max_L) # 维度可以自行修改

    # 训练
    # 完整训练集训练有点慢
    best_train_acc = 0

    dataloader = traindataset(shuffle=True) # 完整训练集
    # dataloader = minitraindataset(shuffle=True) # 用来调试的小训练集
    for i in range(1, max_epoch+1):
        hatys = []
        ys = []
        losss = []
        graph.train()
        X = []
        Y = []
        cnt = 0
        total_count = 0
        for text, label in dataloader:
            #print(np.shape(dataloader))
            x = embedding(text, max_L)
            #print("ok")
            label = np.zeros((1)).astype(np.int32) + label
            X.append(x)

            Y.append(label)
            cnt += 1
            if cnt == batchsize:
                total_count += 1
                if total_count % 50 == 0:
                    print("50 batch done! Total Count:", total_count)
                #print("i:",i,"first:",X[0][0][0])
                X = np.stack(X, 0)
                Y = np.concatenate(Y, 0)
                graph[-1].y = Y
                graph.flush()
                pred, loss = graph.forward(X)[-2:]
                hatys.append(np.argmax(pred, axis=-1))
                ys.append(Y)
                graph.backward()
                graph.optimstep(lr, wd1, wd2)
                losss.append(loss)
                cnt = 0
                X = []
                Y = []

        loss = np.average(losss)
        acc = np.average(np.concatenate(hatys)==np.concatenate(ys))
        print(f"epoch {i} loss {loss:.3e} acc {acc:.4f}")
        if acc > best_train_acc:
            best_train_acc = acc
            with open(save_path, "wb") as f:
                pickle.dump(graph, f)