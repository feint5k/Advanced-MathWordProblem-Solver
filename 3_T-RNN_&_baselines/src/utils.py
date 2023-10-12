
import json                                                                                                                                                                                                                                                                   
import numpy as np
import time

import random
import pickle
import sys

def read_data_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)
    
def write_data_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
        
def save_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
        
def split_by_feilong_23k(data_dict):
    #t_path = "./data/id_ans_test"
    #v_path = "./data/valid_ids.json"
    t_path = "./data/id_ans_test_dolphin"
    v_path = "./data/valid_ids_dolphin.json"
    valid_ids = read_data_json(v_path)
    test_ids = []
    with open(t_path, 'r') as f:
        for line in f:
            test_id = line.strip().split('\t')[0]
            test_ids.append(test_id)
    train_list = []
    test_list = []
    valid_list = []
    for key, value in data_dict.items():
        if key in test_ids:
            test_list.append((key, value))
        elif key in valid_ids:
            valid_list.append((key, value))
        else:
            train_list.append((key, value))
    print (len(train_list), len(valid_list), len(test_list))
    return train_list, valid_list, test_list

def string_2_idx_sen(sen,  vocab_dict):
    #print(sen)
    return [vocab_dict[word] for word in sen]
 
def pad_sen(sen_idx_list, max_len=115, pad_idx=1):                                                                                                                                                                                                                            
    return sen_idx_list + [pad_idx]*(max_len-len(sen_idx_list))

def encoder_hidden_process(encoder_hidden, bidirectional):
    if encoder_hidden is None:
        return None
    if isinstance(encoder_hidden, tuple):
        encoder_hidden = tuple([_cat_directions(h, bidirectional) for h in encoder_hidden])
    else:
        encoder_hidden = _cat_directions(encoder_hidden, bidirectional)
    return encoder_hidden
 
def _cat_directions(h, bidirectional):
    if bidirectional:
        h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
    return h

def post_solver(post_equ):
    stack = [] 
    op_list = ['+', '-', '/', '*', '^']
    for elem in post_equ:
        if elem not in op_list:
            op_v = elem
            if '%' in op_v:
                op_v = float(op_v[:-1])/100.0
            stack.append(str(op_v))
        elif elem in op_list:
            op_v_1 = stack.pop()
            op_v_1 = float(op_v_1)
            op_v_2 = stack.pop()
            op_v_2 = float(op_v_2)
            if elem == '+':
                stack.append(str(op_v_2+op_v_1))
            elif elem == '-':
                stack.append(str(op_v_2-op_v_1))
            elif elem == '*':
                stack.append(str(op_v_2*op_v_1))