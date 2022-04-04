import os
import numpy as np

# set batch_size=1,don't need to pad sentences to the same length
## create corpus
def corpus(char, data_dir='/content/drive/MyDrive/bilstm/all/data'):
    """read data"""
    sentence_word_lists = []
    sentence_tag_lists = []
    with open(os.path.join(data_dir, char+".char.bmes"), 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []
        for line in f:
            if line != '\n':
                word, tag = line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                sentence_word_lists.append(word_list+["<END>"])

                sentence_tag_lists.append(tag_list+["<END>"])

                word_list = []
                tag_list = []
    #sentence_word_lists=sorted(sentence_word_lists,key=lambda x:len(x), reverse=True)
    #sentence_tag_lists=sorted(sentence_tag_lists,key=lambda x:len(x), reverse=True)
    return sentence_word_lists,sentence_tag_lists

#word2id,tag2id
def make_vocab(lists):
    maps={}
    for sentence in lists:
        for word in sentence:
            if word not in maps:
                 maps[word]=len(maps)
    return maps

#get the word index in each sentence
def get_sentence_index(word_lists,tag_lists):
    sentence_word_index_lists=[]
    sentence_tag_index_lists=[]
    for sentence in word_lists:
        sentence_word_index=[]
        for word in sentence:
            if word in word2index:
                index=word2index[word]
            else:
                index=word2index['<UNK>']
            sentence_word_index.append(index)
        sentence_word_index_lists.append(sentence_word_index)
    for sentence_tag in tag_lists:
        sentence_tag_index=[]
        for tag in sentence_tag:
            tag_index=tag2index[tag]
            sentence_tag_index.append(tag_index)
        sentence_tag_index_lists.append(sentence_tag_index)
    return sentence_word_index_lists,sentence_tag_index_lists

#