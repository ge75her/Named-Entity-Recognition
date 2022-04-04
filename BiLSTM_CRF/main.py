from BiLSTM_CRF_data_processing import corpus,make_vocab,get_sentence_index
from BiLSTM_CRF_dataloader import NER
from BiLSTM_CRF_model import BiLSTM_CRF

from torch.utils.data import Dataset, DataLoader
import torch
if __name__=='__main__':
    # create train,dev,test corpus
    # word_list=[['a','b'],['c','d','e']...],tag_list=[['o','B-RACE','M-RACE'],['o','B-NAME','M-NAME'],...]
    train_word_lists, train_tag_lists = corpus('train')
    test_word_lists, test_tag_lists = corpus('test')
    dev_word_lists, dev_tag_lists = corpus('dev')

    # word2index={'a':1,'b':2...},tag2index={'o':1,'B-NAME':2...}
    word2index = make_vocab(train_word_lists)
    tag2index = make_vocab(train_tag_lists)

    # add start stop to list
    word2index['<UNK>'] = len(word2index)
    word2index['<START>'] = len(word2index)
    word2index['<STOP>'] = len(word2index)
    tag2index['<UNK>'] = len(tag2index)
    tag2index['<START>'] = len(tag2index)
    tag2index['<STOP>'] = len(tag2index)

    # the index of word in sentence
    train_sentence_word_index, train_sentence_tag_index = get_sentence_index(train_word_lists, train_tag_lists)
    dev_sentence_word_index, dev_sentence_tag_index = get_sentence_index(dev_word_lists, dev_tag_lists)
    test_sentence_word_index, test_sentence_tag_index = get_sentence_index(test_word_lists, test_tag_lists)


    ##load dataset
    train_dataset = NER(train_sentence_word_index, train_sentence_tag_index)
    dev_dataset = NER(dev_sentence_word_index, dev_sentence_tag_index)
    test_dataset = NER(test_sentence_word_index, test_sentence_tag_index)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    ## model
    # device='cuda:0' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    model = BiLSTM_CRF(tag2idx=tag2index, feature_num=len(word2index), embedding_num=100, hidden_num=512,
                       class_num=len(tag2index), device=device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    ## train process
    for epoch in range(200):
        for sentence, tags in train_loader:
            model.zero_grad()
            sentence = sentence.squeeze()
            tags = tags.squeeze()
            sentence = sentence.to(device)
            tags = tags.to(device)
            loss = model.loss_function(sentence, tags)
            loss.backward()
            optimizer.step()
        if epoch % 50 == 0:
            #print('epoch:{} loss:{:.4}'.format(epoch, loss[0]/len(train_dataset)))
            print(f'epoch:{epoch} loss：{loss[0]/len(train_dataset)}')

    output_path = '/content/drive/MyDrive/bilstm/all/ner_trained_model.cpt'
    torch.save(model, output_path)
    print('=============save model to the path=============\n\n')

    ## predict the tags of the 10th sentence in test dataset
    trained_ner_model = torch.load(output_path)
    with torch.no_grad():
        for i, (j, k) in enumerate(iter(test_loader)):
            j = j.squeeze()
            k = k.squeeze()
            j=j.to(device)
            k=k.to(device)
            if i == 10:
                print('sentence: ', str(j))
                print('tags: ' + str(k))
                print('predict score：' + str(model(j)[0]), 'predict tags：' + str(model(j)[1]))

