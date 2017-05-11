
# coding: utf-8

# In[1]:

import os
import numpy as np
import pandas as pd
from random import shuffle
from tqdm import tqdm
import random

class SKTDataLoader(object):
    """
        What it does:
            - tokenizes the entire dataset and saves
            - creates the vocabulary of word in data
            - indexes the words in vocabulary
            - splits data into train, valid, test sets
            - can return batch_data of specified type(train/valid/test)
            
        TODO: Replace less frequent words with a special word and if new word during test => treat it as the special word
    """
    def __init__(self, input_file_path, output_file_path, batch_size = 64, seq_length = 150, split = [0.70, 0.15, 0.15]):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.base_path = os.path.dirname(self.input_file_path) + '/' + os.path.basename(self.input_file_path).split('.')[0] 
        
        self.input_data = open(input_file_path, 'r').readlines()
        self.output_data = open(output_file_path, 'r').readlines()
       
        self.train_index = 0
        self.cur_index = {}
        self.cur_index['train'] = 0
        self.cur_index['valid'] = 0
        self.cur_index['test'] = 0
        
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.split = split
        
        self.complete_data = self.get_complete_data()
        
        # Get vocab, idx2word, word2idx i.e., all index all words in the data
        self.vocab_dict = self.vocab()
        self.idx2word, self.word2idx = self.index_vocab()
        
        self.go_index = len(self.word2idx) # Index of the word that is to be prepended to make sentences = 'seq_length'
        self.vocab_size = len(self.vocab_dict) + 1 # 1 for filling the sequences which are smaller than 'seq_length'
        
        self.data = self.make_data() # Creates a list self.data which has data which statisfies seq_length condition
        self.data_size = len(self.data)
        self.data_set = self.split_data() # Splits data into train, valid, test sets and creates a dict self.data_set
        
        self.train_size = len(self.data_set['train'])
        self.valid_size = len(self.data_set['valid'])
        self.test_size = len(self.data_set['test'])
        
    def get_complete_data(self):
        """
            Tokenizes every question and save the entire data.
        """
        self.data_path = self.base_path + '_data.npy' 
        if os.path.isfile(self.data_path): # If the indices file is present => load 
            complete_data = np.load(self.data_path)
            print "Loaded complete data"
            return complete_data
            
        complete_data = []  
        for inp, outp in tqdm(zip(self.input_data, self.output_data)):
            inp = inp.split()
            outp = outp.split()
            complete_data.append([inp, outp])
        np.save(self.data_path, complete_data)
        print "Saved and loaded complete data"
        return complete_data
        
    def index_vocab(self):
        """
            Makes dicts(idx2word, word2idx) from vocab_dict
        """
        idx2word = {}
        word2idx = {}
        word2idx_path = self.base_path + '_word2idx.npy'
        idx2word_path = self.base_path + '_idx2word.npy'
        
        if os.path.isfile(word2idx_path) and os.path.isfile(idx2word_path): # If the vocab file is present => load and return only list of words
            word2idx = np.load(word2idx_path).item()
            idx2word = np.load(idx2word_path).item()
            print "Loaded word2idx and idx2word"
            return idx2word, word2idx
        
        for i, word in tqdm(enumerate(self.vocab_dict)):
            word2idx[word] = i
            idx2word[i] = word
        np.save(word2idx_path, word2idx)
        np.save(idx2word_path, idx2word)
        print "Saved and loaded word2idx and idx2word"
        return idx2word, word2idx
                
    def vocab(self):
        """
            Creates a dict vocab, which has all words and their occurence counts.
            Returns the list of all words in the dataset in alphabetical order.
        """
        self.vocab_path = self.base_path + '_vocab.npy'
        if os.path.isfile(self.vocab_path): # If the vocab file is present => load and return only list of words
            vocab = np.load(self.vocab_path)
            vocab_list = vocab.item().keys()
            vocab_list.sort()
            print "Loaded vocab"
            return vocab_list
        vocab = {}
        for inp, outp in tqdm(self.complete_data): # uses all the data to create vocab
            words = inp + outp
            for word in words:
                if word in vocab.keys():
                    vocab[word] += 1
                else:
                    vocab[word] = 1
        np.save(self.vocab_path, vocab)
        vocab_list = vocab.keys()
        vocab_list.sort()
        print "Saved and Loaded vocab"
        return vocab_list
    
    def make_data(self):
        """
            Uses the complete_data and samples data that statisfies seq_length
        """
        data = []
        for inp, outp in self.complete_data:
            if(len(inp) <= self.seq_length  and len(outp) <= self.seq_length):
                data.append([inp, outp])
        return data
    
    def split_data(self):
        """
            Splits data into train, valid, test sets
        """
        # Split the data into train, valid, test using a random data_indices
        self.data_indices_path = self.base_path + '_data_indices_' + str(self.seq_length) + '.npy'
        if os.path.isfile(self.data_indices_path): # If the indices file is present => load 
            self.data_indices = np.load(self.data_indices_path)
        else: # Else create indices now and store
            self.data_indices = range(self.data_size)
            shuffle(self.data_indices)
            np.save(self.data_indices_path, self.data_indices)
            
        # Splitting indices and making train, valid, test data according to their ratio
        self.train_indices = self.data_indices[:int(self.data_size*self.split[0])]
        self.valid_indices = self.data_indices[int(self.data_size*self.split[0]):int(self.data_size*(self.split[0] + self.split[1]))]
        self.test_indices = self.data_indices[int(self.data_size*(self.split[0] + self.split[1])):int(self.data_size*(self.split[0] + self.split[1] + self.split[2]))]
               
        self.train_data = [self.data[x] for x in self.train_indices]
        self.valid_data = [self.data[x] for x in self.valid_indices]
        self.test_data = [self.data[x] for x in self.test_indices]
        
        print "Created dataset"
        return {'train': self.train_data, 'valid': self.valid_data, 'test': self.test_data}
                
    def reset_index(self, data_type='train'):
        self.cur_index[data_type] = 0
        
    def prepend_to_sentence(self, list_of_ids):
        """
            Prepends a list of ids with a id that is not present and reverse the list of input words.
        """
        return [self.go_index]*(self.seq_length - len(list_of_ids)) + list_of_ids[::-1]
    
    def append_to_sentence(self, list_of_ids):
        """
            Appends a list of ids with a id that is not present.
        """
        return list_of_ids + [self.go_index]*(self.seq_length - len(list_of_ids))
      
    def encode_sentence(self, sentence, index):
        """
            Takes in a list of words and encodes them using word2id.
        """
        if index == 0:
            return self.prepend_to_sentence([self.word2idx[x] for x in sentence])
        return self.append_to_sentence([self.word2idx[x] for x in sentence])
    
    def encode_batch(self, batch_data):
        batch_inp = []
        batch_outp = []
        for pair in batch_data:
            batch_inp.append(self.encode_sentence(pair[0], 0))
            batch_outp.append(self.encode_sentence(pair[1], 1))
        return batch_inp, batch_outp
    
    def next_batch(self, data_type='train'):
        """
            Returns batch_data of size 'batch_size' and of corresponding 'data_type'
        """
        stop = False
        if data_type == 'train':
            if(self.cur_index[data_type] + self.batch_size > self.train_size):
                self.reset_index(data_type = data_type)
                stop = True
        if data_type == 'test':
            if(self.cur_index[data_type] + self.batch_size > self.test_size):
                self.reset_index(data_type = data_type)
                stop = True
        if data_type == 'valid':
            if(self.cur_index[data_type] + self.batch_size > self.valid_size):
                self.reset_index(data_type = data_type)
                stop = True
        batch_data = self.data_set[data_type][self.cur_index[data_type] : self.cur_index[data_type] + self.batch_size]
        batch_data = self.encode_batch(batch_data)
        self.cur_index[data_type] += self.batch_size
                
        return np.swapaxes(np.array(batch_data[0]), 0, 1), np.swapaxes(np.array(batch_data[1]), 0, 1)
    
    def random_batch(self, data_type='train'):

        if data_type == 'train':
            temp_index = random.randint(0, self.train_size-self.batch_size-1)
        if data_type == 'test':
            temp_index = random.randint(0, self.test_size-self.batch_size-1)
        if data_type == 'valid':
            temp_index = random.randint(0, self.valid_size-self.batch_size-1)
        
        batch_data = self.data_set[data_type][temp_index : temp_index + self.batch_size]
        batch_data = self.encode_batch(batch_data)
               
        return np.swapaxes(np.array(batch_data[0]), 0, 1), np.swapaxes(np.array(batch_data[1]), 0, 1)
