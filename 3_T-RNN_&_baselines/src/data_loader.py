
import json                                                                                                                                                                                                                                                                   
import numpy as np
import time

import random
import pickle
import sys

from utils import *

class DataLoader:
    def __init__(self, dataset):#, dataset, math23k_vocab_list, math23k_decode_list):
 
        self.dataset = dataset
 
        math23k_vocab_list = ['PAD','<S>', 'EOS', '1', 'PI']
        math23k_decode_list = ['temp_0','temp_1','temp_2','temp_3','temp_4','temp_5','temp_6','temp_7',\
                              'temp_8', 'temp_9', 'temp_10', 'temp_11', 'temp_12','temp_13',#'temp_14', 
				'1', 'PI', 'PAD', '<S>','EOS']

        #math23k_decode_list = ['temp_0','temp_1','temp_2','temp_3','1', 'PI', 'PAD', '<S>','EOS']
        for k, v in dataset.items():
            for elem in v['template_text'].split(' '):
                if elem not in math23k_vocab_list:
                    math23k_vocab_list.append(elem)
 
        self.train_list, self.valid_list, self.test_list = split_by_feilong_23k(dataset)
        self.data_set = dataset
        self.vocab_list = math23k_vocab_list
        self.vocab_dict = dict([(elem, idx) for idx, elem in enumerate(self.vocab_list)])
        self.vocab_len = len(self.vocab_list)
 
        self.decode_classes_list = math23k_decode_list
        self.decode_classes_dict = dict([(elem, idx) for idx, elem in enumerate(self.decode_classes_list)])
        self.decode_classed_len = len(self.decode_classes_list)
 
        self.decode_emb_idx = [self.vocab_dict[elem] for elem in self.decode_classes_list]
    
        


    def __len__(self):
        return self.data_size
    
    def _data_batch_preprocess(self, data_batch):
        batch_encode_idx = []
        batch_encode_len = []
        batch_encode_num_pos = []
        
        batch_decode_idx = []
        batch_decode_emb_idx = []
        batch_decode_len = []
 
        batch_idxs = []
        batch_text = []
        batch_num_list = []
        batch_solution = []
        
        batch_post_equation = []
        
        batch_gd_tree = []
        #batch_mask = []
        
        #dict_keys(['numtemp_order', 'index', 'post_template', 'ans', 'num_list', 'template_text', 'mid_template', 'expression', 'text'])
        for elem in data_batch:
            idx = elem[0]
            encode_sen = elem[1]['template_text'][:]
            encode_sen_idx = string_2_idx_sen(encode_sen.strip().split(' '), self.vocab_dict)
            
            batch_encode_idx.append(encode_sen_idx)
            batch_encode_len.append(len(encode_sen_idx))
            batch_encode_num_pos.append(elem[1]['num_position'][:])
            
            decode_sen = elem[1]['numtemp_order'][:]
            #print (decode_sen)
            decode_sen.append('EOS')
            decode_sen_idx = string_2_idx_sen(decode_sen, self.decode_classes_dict)
            decode_sen_emb_idx = [self.decode_emb_idx[elem] for elem in decode_sen_idx]