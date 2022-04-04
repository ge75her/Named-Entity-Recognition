import torch.nn as nn
class BiLSTM(nn.Module):
    #feature_num=word num, embedding_num=100,class_num=tag num
    def __init__(self,feature_num,embedding_num,hidden_num,class_num,bi=True):
        super(BiLSTM,self).__init__()
        self.embedding=nn.Embedding(feature_num,embedding_num)
        self.lstm=nn.LSTM(embedding_num,hidden_num,batch_first=True,bidirectional=bi)#batch_first:[batch,seq,embedding_num]
        if bi:
            self.linear1=nn.Linear(hidden_num*2,hidden_num)
            self.linear2=nn.Linear(hidden_num,class_num)
        else:
            self.linear1=nn.Linear(hidden_num,hidden_num/2)
            self.linear2=nn.Linear(hidden_num/2,class_num)
    def forward(self,x):
        x=self.embedding(x)
        x,hidden=self.lstm(x)
        x=self.linear1(x)
        x=self.linear2(x)
        return x