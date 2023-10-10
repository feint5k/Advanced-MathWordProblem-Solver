
import json																																	  
import numpy as np
import time

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import logging
import random
import pickle
import sys

from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaseRNN(nn.Module):
	def __init__(self, vocab_size, emb_size, hidden_size, input_dropout_p, dropout_p, \
			  n_layers, rnn_cell_name):
		super(BaseRNN, self).__init__()
		self.vocab_size = vocab_size
		self.emb_size = emb_size
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.input_dropout_p = input_dropout_p
		self.input_dropout = nn.Dropout(p=input_dropout_p)
		self.rnn_cell_name = rnn_cell_name
		if rnn_cell_name.lower() == 'lstm':
			self.rnn_cell = nn.LSTM
		elif rnn_cell_name.lower() == 'gru':
			self.rnn_cell = nn.GRU
		else:
			raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell_name))
		self.dropout_p = dropout_p
	 
	def forward(self, *args, **kwargs):
		raise NotImplementedError()

class EncoderRNN(BaseRNN):
	def __init__(self, vocab_size, embed_model, emb_size=100, hidden_size=128, \
		 input_dropout_p=0, dropout_p=0, n_layers=1, bidirectional=False, \
		 rnn_cell=None, rnn_cell_name='gru', variable_lengths_flag=True):
		super(EncoderRNN, self).__init__(vocab_size, emb_size, hidden_size,
			  input_dropout_p, dropout_p, n_layers, rnn_cell_name)
		self.variable_lengths_flag = variable_lengths_flag
		self.bidirectional = bidirectional
		self.embedding = embed_model
		if rnn_cell == None:
			self.rnn = self.rnn_cell(emb_size, hidden_size, n_layers,
					 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)
		else:
			self.rnn = rnn_cell
 
	def forward(self, input_var, input_lengths=None):
		embedded = self.embedding(input_var)
		embedded = self.input_dropout(embedded)
		#pdb.set_trace()
		if self.variable_lengths_flag:
			embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
		output, hidden = self.rnn(embedded)
		if self.variable_lengths_flag:
			output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
		return output, hidden

class Attention_1(nn.Module):
	def __init__(self, input_size, output_size):
		super(Attention_1, self).__init__()
		self.linear_out = nn.Linear(input_size, output_size)
		#self.mask = Parameter(torch.ones(1), requires_grad=False)
	 
	#def set_mask(self, batch_size, input_length, num_pos):
	#	 self.mask = self.mask.repeat(input_length).unsqueeze(0).repeat(batch_size, 1)
	#	 for mask_i in range(batch_size):
	#		 self.mask[mask_i][num_pos[mask_i]] = 1 
	def init_mask(self, size_0, size_1, input_length):
		mask = Parameter(torch.ones(1), requires_grad=False)
		mask = mask.repeat(size_1).unsqueeze(0).repeat(size_0, 1)
		#for i in range(input_length)
		input_index = list(range(input_length))
		for i in range(size_0):
			mask[i][input_index] = 0
		#print (mask)
		mask = mask.byte()
		#mask = mask.cuda()
		mask = mask.to(device)
		return mask
		
	
	 
	def _forward(self, output, context, input_lengths, mask):
		'''
		output: len x hidden_size
		context: num_len x hidden_size
		input_lengths: torch scalar
		'''
		#print (output.size()) torch.Size([5, 256])
		#print (.size()) torch.Size([80, 256])
		#print (input_lengths)
		attn = torch.matmul(output, context.transpose(1,0))
		#print (attn.size()) 0 x 1
		attn.data.masked_fill_(mask, -float('inf'))
		attn = F.softmax(attn, dim=1)
		#print (attn)
		mix = torch.matmul(attn, context)
		#print ("mix:", mix)
		#print ("output:", output)
		combined = torch.cat((mix, output), dim=1)
		#print ("combined:",combined)
		output = F.tanh(self.linear_out(combined))
		
		#print ("output:",output)
		#print ("------------")
		#print ()
		return output, attn
	
	
	
	
	def forward(self, output, context, num_pos, input_lengths):
		'''
		output: decoder,  (batch, 1, hiddem_dim2)
		context: from encoder, (batch, n, hidden_dim1)
		actually, dim2 == dim1, otherwise cannot do matrix multiplication 
		 
		'''
		batch_size = output.size(0)
		hidden_size = output.size(2)																														  
		input_size = context.size(1)
		#print ('att:', hidden_size, input_size)
		#print ("context", context.size())
		
		attn_list = []
		mask_list = []
		output_list = []
		for b_i in range(batch_size):
			per_output = output[b_i]
			per_num_pos = num_pos[b_i]
			#print(context, num_pos)
			current_output = per_output[per_num_pos]
			per_mask = self.init_mask(len(per_num_pos), input_size, input_lengths[b_i])
			mask_list.append(per_mask)
			#print ("current_context:", current_context.size())
			per_output, per_attn = self._forward(current_output, context[b_i], input_lengths[b_i], per_mask)
			#for p_j in range(len(per_num_pos)):
			#	 current_context = per_context[per_num_pos[p_j]]
			#	 print ("c_context:", current_context.size())
			output_list.append(per_output)
			attn_list.append(per_attn)
			
			
		
		#self.set_mask(batch_size, input_size, num_pos)
		# (b, o, dim) * (b, dim, i) -> (b, o, i)
		'''
		attn = torch.bmm(output, context.transpose(1,2))
		if self.mask is not None: