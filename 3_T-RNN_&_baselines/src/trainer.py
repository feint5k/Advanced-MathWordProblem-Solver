
from utils import *
from model import *
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer():
	def __init__(self, data_loader, params):
		self.data_loader = data_loader
		self.params = params
		self.train_len = len(data_loader.train_list)
		self.valid_len = len(data_loader.valid_list)
		self.test_len = len(data_loader.test_list)
		#self.optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, dampening=0.0)
		#self.pg_seq = dict(read_data_json("./data/pg_seq_norm_0828.json")) #dict
		self.pg_seq = dict(read_data_json("./data/pg_norm_test_dolphin.json")) #dict
	
	def _train_batch_recur(self, model, batch_encode_pad_idx, batch_encode_num_pos, batch_encode_len, batch_gd_tree):
		batch_encode_pad_idx_tensor = torch.LongTensor(batch_encode_pad_idx)
		batch_encode_tensor_len = torch.LongTensor(batch_encode_len)
		
		batch_encode_pad_idx_tensor = batch_encode_pad_idx_tensor.to(device)#cuda()
		batch_encode_tensor_len = batch_encode_tensor_len.to(device)#cuda()
		#print ("batch_encode_num_pos",batch_encode_num_pos)
		b_pred, b_loss, b_count, b_acc_e, b_acc_e_t, b_acc_i = model(batch_encode_pad_idx_tensor, batch_encode_tensor_len, \
			  batch_encode_num_pos, batch_gd_tree)
		self.optimizer.zero_grad()
		#print (b_loss)
		b_loss.backward(retain_graph=True)
		clip_grad_norm_(model.parameters(), 5, norm_type=2.)
		self.optimizer.step()
		return b_pred, b_loss.item(), b_count, b_acc_e, b_acc_e_t, b_acc_i

	def _test_recur(self, model, data_list):
		batch_size = self.params['batch_size']
		data_generator = self.data_loader.get_batch(data_list, batch_size)
		test_pred = []
		test_count = 0
		test_acc_e = []
		test_acc_e_t = []
		test_acc_i = []
		for batch_elem in data_generator:
			batch_encode_idx = batch_elem['batch_encode_idx']
			batch_encode_pad_idx = batch_elem['batch_encode_pad_idx']
			batch_encode_num_pos = batch_elem['batch_encode_num_pos']
			batch_encode_len = batch_elem['batch_encode_len']

			batch_decode_idx = batch_elem['batch_decode_idx']

			batch_gd_tree = batch_elem['batch_gd_tree']
			
			batch_encode_pad_idx_tensor = torch.LongTensor(batch_encode_pad_idx)
			batch_encode_tensor_len = torch.LongTensor(batch_encode_len)

			batch_encode_pad_idx_tensor = batch_encode_pad_idx_tensor.to(device)#cuda()
			batch_encode_tensor_len = batch_encode_tensor_len.to(device)#cuda()
			
			b_pred, b_count, b_acc_e, b_acc_e_t, b_acc_i = model.test_forward_recur(batch_encode_pad_idx_tensor, batch_encode_tensor_len, \
			  batch_encode_num_pos, batch_gd_tree)
			
			test_pred+= b_pred
			test_count += b_count
			test_acc_e += b_acc_e
			test_acc_e_t += b_acc_e_t
			test_acc_i += b_acc_i
		return test_pred, test_count, test_acc_e, test_acc_e_t, test_acc_i