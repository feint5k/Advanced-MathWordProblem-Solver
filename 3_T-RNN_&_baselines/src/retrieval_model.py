
######################################################################
# File: retrieval_model.py
# Author: Vishal Dey
# Created on: 11 Dec 2019
#######################################################################
'''
 Synopsis: Create w2v for corresponding string description of each problem
	Reads in pretrianed Word2vec vectors and add each word vector to obtain
	phrase vectors.
	Either compute TF-IDF / w2v based cosine similarity
'''

import os
import json
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from copy import deepcopy

# for reproducibility
np.random.choice(123)


# read json file
def read_json(fname):
	with open(os.path.join('./data/', fname), 'r') as fp:
		return json.load(fp)


# compute TF-iDF based most similar problem
def find_similar_tfidf(tfidf_matrix, tfidf_vector):
	cosine_similarities = linear_kernel(tfidf_vector, tfidf_matrix).flatten()
	related_docs_indices = [i for i in cosine_similarities.argsort()[::-1]]
	#print(related_docs_indices[0], cosine_similarities[related_docs_indices[0]])
	return related_docs_indices[0]

# compute w2v cosine based most similar problem
def find_similar_w2v(w2vmatrix, w2vemb, query, EMB_DIM, minv=0, maxv=1):
	query_emb = []
	for word in query.split():
		if word in w2vemb:
			query_emb.append(np.array(list(map(float, w2vemb[word]))))
		else:
			query_emb.append(np.random.uniform(minv, maxv, EMB_DIM))
	
	cosine_similarities = linear_kernel(query_emb, w2vmatrix).flatten()
	related_docs_indices = [i for i in cosine_similarities.argsort()[::-1]]
	print(related_docs_indices)
	return related_docs_indices[0]


# load w2v problem
def load_w2v(w2v_file):
	EMB_DIM = 0
	w2v_emb = {}
	minv = sys.float_info.max
	maxv = sys.float_info.min

	with open(os.path.join('./w2v', w2v_file), 'r') as fp:
		EMB_DIM = int(fp.readline().split()[1])
		for line in fp.readlines():
			tmp = line.split()
			tmp = list(map(float, tmp[1:]))