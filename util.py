#    Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
#    Written by Nikolaos Pappas <nikolaos.pappas@idiap.ch>,
#
#    This file is part of mhan.
#
#    mhan is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License version 3 as
#    published by the Free Software Foundation.
#
#    mhan is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with mhan. If not, see http://www.gnu.org/licenses/

import os
import sys
import pickle
import json, gzip
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from keras.utils.np_utils import to_categorical

LABB = {'english': 'en', 'german': 'de', 'spanish': 'es',
	'portuguese': 'pt', 'ukrainian': 'uk', 'russian': 'ru',
	'arabic': 'ar', 'persian': 'fa' }

def load_data(path=None):
	"""Load and return the specified dataset."""
	h =  json.load(open(path))
	print ("\t%s" % (path)).ljust(60) + "OK"
	return h['X_ids'], h['Y_ids'], h['label_ids']

def load_word_vectors(language, path):
	"""Function to load pre-trained word vectors."""
	print "[*] Loading %s word vectors..." % language
	wvec, vocab = {}, {}
	embeddings = pickle.load(gzip.open(path+".gz"))
	wvec[language] = embeddings[1]
	vocab[language] = list(embeddings[0])
	print ("\t%s" % (path)).ljust(60) + "OK"
	return wvec, vocab

def load_vectors(wvec, labels, x_idxs, y_idxs, wpad, spad):
	"""Load word vectors for a given sequence and apply zero-padding
	   according to the pre-defined limits."""
	X_vecs, Y_labels, total, wdim = [],  [], spad*wpad, wvec[0].shape[0]
	for idx, x_idx_set in enumerate(x_idxs):
		x_vec =[]
		for j,x in enumerate(x_idx_set):
		  	vecs = wvec[x[:wpad]]
		  	zeros = np.zeros((wpad, wdim))
		  	zeros[0:len(vecs)] = vecs
			if j == 0:
				x_vec = zeros
			else:
				x_vec = np.vstack([x_vec, zeros])
		if x_vec.shape[0] < total:
			szeros = np.zeros((total - x_vec.shape[0], wdim))
			x_vec = np.vstack([x_vec,szeros])
		else:
			x_vec = x_vec[:total,:]
		y_cats =  np.sum(to_categorical(y_idxs[idx], num_classes=len(labels)),axis=0)
		y_cats[y_cats>1] = 1
		X_vecs.append(x_vec)
		Y_labels.append(y_cats)
	return X_vecs, Y_labels

def pick_best(dev_path):
	""" Pick the best model according to its validation score in the
	    specified experiment folder path. """
	fs = [float(open(dev_path+fn).read().split(' ')[2]) for fn in os.listdir(dev_path) if fn.find('val')>-1]
	weights = [fn for fn in os.listdir(dev_path) if fn.find('weights')>-1]
	best_idx = fs.index(max(fs))
	epoch_num = int(weights[best_idx].split('_')[1].split('-')[0])
	print "[*] Loading best model (e=%d, f1=%.3f)..." % (epoch_num, fs[best_idx])
	print ("\t%s" % (dev_path+weights[best_idx]) ).ljust(60) + "OK"
	return epoch_num, dev_path+weights[best_idx]

def export(lang, lang_idx, source_idx, epreds, watts, satts, XT_ids, YT_ids, vocabs, labels, top_k=20):
	""" Function which returns a dictionary with the top-k predictions and
	    attention scores of a given model on a given test set, along with
	    the corresponding texts and their gold labels. """
	out = {}
	for i, x in enumerate(XT_ids[lang_idx]):
		text, tags, gold_tags = [], [], []
		for j, xi in enumerate(x):
			text.append([vocabs[lang_idx][wid] for wid in xi])
		for y in YT_ids[lang_idx][i]:
			labwords = labels[lang_idx][y]
			words = [vocabs[lang_idx][int(wid)] for wid in labwords.split('_')]
			gold_tags.append(' '.join(words))
		top_ids = np.argsort(epreds[i])[::-1]
		for y in top_ids[:top_k]:
			labwords = labels[source_idx][y]
			words = [vocabs[source_idx][int(wid)] for wid in labwords.split('_')]
			tags.append([' '.join(words), str(epreds[i][y])] )
		out["%s_%d" % (	LABB[lang], i)] = {'text': text,
				  'watts': watts[i].tolist(),
				  'satts': satts[i].tolist(),
				  'gold_tags': gold_tags,
				  'tags': tags}
	return out
