# NER Problems

## Data preprocessing
import os
import numpy as np
# create corpus
def corpus(char, data_dir="data"):
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

#pad sentence to the same size in order to fed in LSTM model
def pad_sentence(word_sentence_lists,tag_sentence_lists,max_len):
    word_pad_sentence_list=[]
    for sentence in word_sentence_lists:
        length=len(sentence)
        if length>=max_len:
            pad_sentence=sentence[:max_len]
        else:
            pad_sentence=sentence
            pad_len=max_len-length
            for _ in range(pad_len):
                pad_sentence.append(word2index['<PAD>'])
        word_pad_sentence_list.append(pad_sentence)

    tag_pad_sentence_list=[]
    for sentence in tag_sentence_lists:
        length=len(sentence)
        if length>=max_len:
            pad_sentence=sentence[:max_len]
        else:
            pad_sentence=sentence
            pad_len=max_len-length
            for _ in range(pad_len):
                pad_sentence.append(tag2index['<PAD>'])
             #
        tag_pad_sentence_list.append(pad_sentence)
    return word_pad_sentence_list,tag_pad_sentence_list

