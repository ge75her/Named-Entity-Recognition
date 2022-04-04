from BiLSTM_data_processing import corpus,make_vocab,get_sentence_index,pad_sentence
from BiLSTM_dataloader import NER
from BiLSTM import BiLSTM

from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
if __name__=='__main__':
    ## create train,dev,test corpus
    # word_list=[['a','b'],['c','d','e']...],tag_list=[['o','B-RACE','M-RACE'],['o','B-NAME','M-NAME'],...]
    train_word_lists, train_tag_lists = corpus('train')
    test_word_lists, test_tag_lists = corpus('test')
    dev_word_lists, dev_tag_lists = corpus('dev')
    # word2index={'a':1,'b':2...},tag2index={'o':1,'B-NAME':2...}
    word2index = make_vocab(train_word_lists)
    tag2index = make_vocab(train_tag_lists)

    # add unk pad to list
    word2index['<UNK>'] = len(word2index)
    word2index['<PAD>'] = len(word2index)
    word2index['<START>'] = len(word2index)
    tag2index['<UNK>'] = len(tag2index)
    tag2index['<PAD>'] = len(tag2index)
    tag2index['<START>'] = len(tag2index)
    # the index of word in sentence
    train_sentence_word_index, train_sentence_tag_index = get_sentence_index(train_word_lists, train_tag_lists)
    dev_sentence_word_index, dev_sentence_tag_index = get_sentence_index(dev_word_lists, dev_tag_lists)
    test_sentence_word_index, test_sentence_tag_index = get_sentence_index(test_word_lists, test_tag_lists)
    # pad sentence with max_len=50
    pad_train_word, pad_train_tag = pad_sentence(train_sentence_word_index, train_sentence_tag_index, 50)
    pad_dev_word, pad_dev_tag = pad_sentence(dev_sentence_word_index, dev_sentence_tag_index, 50)
    pad_test_word, pad_test_tag = pad_sentence(test_sentence_word_index, test_sentence_tag_index, 50)
    # =pad_sentence(train_sentence_tag_index,50,word=False)

    ##load dataset
    train_dataset = NER(pad_train_word, pad_train_tag)
    dev_dataset = NER(pad_dev_word, pad_dev_tag)
    test_dataset = NER(pad_test_word, pad_test_tag)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(dev_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    feature_num = len(word2index)
    embedding_num = 100
    hidden_num = 512
    class_num = 50
    model = BiLSTM(feature_num, embedding_num, hidden_num, class_num)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    cir = nn.CrossEntropyLoss(ignore_index=word2index['<PAD>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(30):
        model.train()
        train_loss = 0.0
        for data, tag in train_loader:
            optimizer.zero_grad()
            data = data.to(device)
            tag = tag.to(device)
            train_pred = model(data)

            batch_loss = cir(train_pred, tag)
            batch_loss.backward()
            optimizer.step()
            train_loss += batch_loss.item()

        print("epcoh", epoch, f"loss:{train_loss:.2f}")

