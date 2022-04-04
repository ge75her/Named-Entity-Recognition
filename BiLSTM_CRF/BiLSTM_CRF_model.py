import torch
import torch.nn as nn
import torch.autograd as autograd

def to_scalar(var):
    return var.view(-1).data.tolist()[0]
def argmax(vec):
    _,idx=torch.max(vec,1)
    return to_scalar(idx)

def log_sum_exp(vec):
    '''
    state matrix[i][j]: i表示上个状态，j表示当前状态
    [[1 3 9]
    [2 9 1]
    [3 4 7]]: 第二列表示当前状态到达2状态的三种可能的来源
    score(2)=log(exp(9)+exp(1)+exp(7))
    若有很大的数，直接计算会有溢出的问题，用log([exp(smat[0]-vmax)+exp(smat[1]-vmax)+exp(smat[2]-vmax)])+vmax 解决
    最终return [score0,score1,score2]
    '''
    max_score=vec[0,argmax(vec)]
    max_score_broadcast=max_score.view(1,-1).expand(1,vec.size()[1])
    return max_score+torch.log(torch.sum(torch.exp(vec-max_score_broadcast)))


class BiLSTM_CRF(nn.Module):
    def __init__(self, tag2idx, feature_num, embedding_num, hidden_num, class_num, device):
        super(BiLSTM_CRF, self).__init__()
        self.tag2idx = tag2idx
        self.feature_num = feature_num
        self.embedding_num = embedding_num
        self.hidden_num = hidden_num
        self.class_num = class_num
        self.device = device

        self.embedding = nn.Embedding(feature_num, embedding_num)
        self.lstm = nn.LSTM(embedding_num, hidden_num // 2, num_layers=1, bidirectional=True)  # bilstm,hidden_num//2
        self.linearlstm = nn.Linear(hidden_num, class_num)  # emission score
        # define a random init transitions matrix (label to label)
        self.transitions = nn.Parameter(torch.randn(class_num, class_num))
        # these two statements enforce the constraint that we never transfer
        self.transitions.data[self.tag2idx['<START>'], :] = self.transitions.data[:, self.tag2idx['<END>']] = -10000
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # random init hidden layers:(h0,c0)
        # shape:(num_layers*num_dir,batch_size,hidden_size)
        return (autograd.Variable(torch.randn(2, 1, self.hidden_num // 2).to(device)),
                autograd.Variable(torch.randn(2, 1, self.hidden_num // 2)).to(device))

    def loss_function(self, words, tags):
        emission = self._get_lstm_features(words)
        up = self._real_path_score(emission, tags)
        down = self._total_path_score(emission)
        score = down - up
        return score

    def _get_lstm_features(self, sentence):

        self.hidden = self.init_hidden()
        embeds = self.embedding(sentence)
        # embeds=embeds.squeeze()

        # lstm input shape；(seq_len,batch_size=1,embedding_size),emdeding shape:(seq_len,embedding_num)
        embeds = embeds.unsqueeze(1)
        '''
Outputs: output, (h_n, c_n)
output (seq_len, batch, hidden_size * num_directions): tensor containing the output features (h_t) from the last layer of the RNN for each t. 
If a torch.nn.utils.rnn.PackedSequence has been given as the input, the output will also be a packed sequence.
h_n (num_layers * num_directions, batch, hidden_size): tensor containing the hidden state for t=seq_len,最后一个单词的hidden state
c_n (num_layers * num_directions, batch, hidden_size): tensor containing the cell state for t=seq_len,最后一个单词的cell state
        '''

        lstm_out, self.hidden = self.lstm(embeds, self.hidden)  # out shape:(seq_len,batch=1,hidden_num)
        lstm_out = lstm_out.view(len(sentence), self.hidden_num)  # shape:(seq_len,hidden_num),去掉batch_dim
        lstm_feat = self.linearlstm(lstm_out)  # shape: (1,class_num)
        return lstm_feat

    def _real_path_score(self, feats, tags):
        '''
        tags=start+tag2index+end
        '''
        score = autograd.Variable(torch.Tensor([0])).to(device)
        tags = torch.cat([torch.LongTensor([self.tag2idx['<START>']]).to(device), tags])  # concat start+sentence
        for i, feat in enumerate(feats):
            # score=emission_score+transition_score
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]  # 多了一个start label， i+1
        score = score + self.transitions[self.tag2idx['<END>'], tags[-1]]
        return score

    def _total_path_score(self, feats):
        init_alphas = torch.Tensor(1, self.class_num).fill_(-10000.)
        init_alphas[0][self.tag2idx['<START>']] = 0.  # start tag has all the scores
        forward_var = autograd.Variable(init_alphas).to(device)

        for feat in feats:
            alphas_t = []
            for next_tag in range(self.class_num):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.class_num)  # shape:(1,50)
                trans_score = self.transitions[next_tag].view(1, -1)  # shape: (1,50)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).unsqueeze(0))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag2idx['<STOP>']]
        alpha = log_sum_exp(terminal_var)  # 0-dim tensor
        return alpha

    def _viterbi(self, feats):
        backpointers = []
        # init
        init_vvars = torch.Tensor(1, self.class_num).fill_(-10000.)
        # init_vvars = torch.Tensor((1, self.class_num), -10000.)
        init_vvars[0][self.tag2idx['<START>']] = 0

        forward_var = autograd.Variable(init_vvars)
        for feat in feats:
            # keep index to get the best path
            bptrs_t = []
            # keep probaility in the path
            viterbivars_t = []

            for next_tag in range(self.class_num):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # move to stop_tag
        terminal_var = forward_var + self.transitions[self.tag2idx['<STOP>']]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # backward get the best path
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        print(best_path)
        # pop start_tag
        start = best_path.pop()
        print(start)
        assert start == self.tag2idx['<START>']
        best_path.reverse()
        return path_score, best_path

    def forward(self, sentence):
        lstm_feats = self._get_lstm_features(sentence)
        score, tag_seq = self._viterbi(lstm_feats)
        return score, tag_seq
