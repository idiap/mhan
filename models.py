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
import os, sys
import numpy as np
import time, theano, json, keras
from util import load_vectors
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers.core import Lambda, Reshape
if keras.__version__[0] == '1':
    from keras.layers import Merge
else:
    from keras.layers.merge import Multiply, Concatenate
from keras.layers import Input, TimeDistributed, Dense, GRU
from keras.layers import Permute, RepeatVector, Flatten, Activation
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class MHAN:
	"""
	Class which contains all the necessary functions to create and train
	multilingual hierarchical attention neural networks based on three
	component sharing configurations:
		1. Sharing encoders at both levels
		2. Sharing attention at both levels
		3. Sharing encoders and attention at both levels
	"""
	def __init__(self, args):
		self.args = args
		self.args['wpad'] = self.args['swpad']*self.args['spad']
		self.single_language = len(self.args['languages']) == 1
		self.attention_mode = self.args['enc'].find('att') > -1
		self.args["languages"] = [args["source"]] if args["source"] else args["languages"]

	def build_encoders(self):
		""" Builds functions needed for the word-level and sentence-level
			encoders and returns them in a dictionary. """
		backsent_enc, backdoc_enc = None, None
		if self.args['enc'] in ["dense","attdense"]:
			sent_enc = TimeDistributed(Dense(self.args['sdim'],
							input_shape=(self.args['wpad'], self.args['wdim']),
							activation=self.args['act']),
							input_shape=(self.args['wpad'], self.args['wdim']))
			doc_enc = TimeDistributed(Dense(self.args['ddim'],
							input_shape=(self.args['spad'], self.args['wdim']),
							activation=self.args['act']),
							input_shape=(self.args['spad'], self.args['wdim']))
		elif self.args['enc'] in ["gru", "attgru"]:
			sent_enc = GRU(self.args['sdim'],
					input_shape=(self.args['wpad'], self.args['wdim']),
					activation=self.args['gruact'],
					return_sequences=True)
			doc_enc = GRU(self.args['ddim'],
				      input_shape=(self.args['spad'], self.args['wdim']),
				      activation=self.args['gruact'],
				      return_sequences=True)
		elif self.args['enc'] in ["bigru", "attbigru"]:
			sent_enc = GRU(self.args['sdim'],
						   input_shape=(self.args['wpad'], self.args['wdim']),
						   activation=self.args['gruact'],
						   return_sequences=True)
			backsent_enc = GRU(self.args['sdim'],
					   input_shape=(self.args['wpad'], self.args['wdim']),
					   go_backwards=True,
					   activation=self.args['gruact'],
					   return_sequences=True)
			doc_enc = GRU(self.args['ddim'],
				      input_shape=(self.args['spad'], self.args['wdim']*2),
				      activation=self.args['gruact'],
				      return_sequences=True)
			backdoc_enc = GRU(self.args['ddim'],
					  input_shape=(self.args['spad'], self.args['wdim']*2),
					  go_backwards=True,
					  activation=self.args['gruact'],
					  return_sequences=True)
		encoders = {'sent_enc': sent_enc,
			    'doc_enc': doc_enc,
			    'backsent_enc': backsent_enc,
			    'backdoc_enc': backdoc_enc}
		return encoders

	def build_attention(self, lang):
		""" Builds functions needed for the word-level and sentence-level
		    attention mechanisms. """
		bigru = self.args['enc'].find('bigru') > -1
		hsdim = self.args['sdim']*2 if bigru else self.args['sdim']
		hddim = self.args['ddim']*2 if bigru else self.args['ddim']
		sent_enc = TimeDistributed(Dense(hddim, activation=self.args['act']))
		sent_context = TimeDistributed(Dense(1, activation=self.args['act']))
		word_enc = TimeDistributed(Dense(hsdim, activation=self.args['act']))
		word_context = TimeDistributed(Dense(1, activation=self.args['act']))
		submax_sent = Lambda(self.submax, output_shape=self.submax_output)
		submax_word = Lambda(self.submax, output_shape=self.submax_output)
		softmax_sent = Activation(activation="softmax", name="%s_satt" % lang)
		softmax_word = Activation(activation="softmax", name="%s_watt" % lang)
		reshape_word = Reshape((self.args['spad'],self.args['swpad']))
		repeat_word = RepeatVector(hsdim)
		repeat_sent = RepeatVector(hddim)
		permute_sent = Permute((2,1))
		permute_word = Permute((2,1))
		flatten_sent = Flatten()
		flatten_word = Flatten()
		flatten_word_after = Flatten()
		attention = {'sent_enc': sent_enc,
			     'sent_context': sent_context,
			     'word_enc': word_enc,
			     'word_context': word_context,
			     'flatten_sent': flatten_sent,
			     'flatten_word': flatten_word,
			     'flatten_word_after': flatten_word_after,
			     'submax_sent': submax_sent,
			     'submax_word': submax_word,
			     'softmax_sent': softmax_sent,
			     'softmax_word': softmax_word,
			     'repeat_sent': repeat_sent,
			     'repeat_word': repeat_word,
			     'permute_sent': permute_sent,
			     'permute_word': permute_word,
			     'reshape_word': reshape_word}
		return attention

	def word_attention(self, forward_words, attention):
		""" Compute word-level attention scores and return attented
		    word vectors for the whole word sequence. """
		hdim = forward_words._keras_shape[2]
		embedded_words = attention['word_enc'](forward_words)
		attented_words = attention['word_context'](embedded_words)
		weights = attention['flatten_word'](attented_words)
		reshaped = attention['reshape_word'](weights)
		submaxed = attention['submax_word'](reshaped)
		weights = attention['softmax_word'](submaxed)
		weights = attention['flatten_word_after'](weights)
		weights = attention['repeat_word'](weights)
		weights = attention['permute_word'](weights)
		if keras.__version__[0] == '1': 
		    return Merge([weights,forward_words], mode='mul')
		else:
		    return Multiply()([weights, forward_words])

	def sentence_attention(self, forward_doc, attention):
		""" Compute sentence-level attention scores and return attented
		    word vectors for the whole sentence sequence. """
		hdim = forward_doc._keras_shape[2]
		embedded_sents = attention['sent_enc'](forward_doc)
 		attented_sents = attention['sent_context'](embedded_sents)
		sent_weights = attention['flatten_sent'](attented_sents)
		submaxed = attention['submax_sent'](sent_weights)
		sent_weights = attention['softmax_sent'](submaxed)
		sent_weights = attention['repeat_sent'](sent_weights)
		sent_weights = attention['permute_sent'](sent_weights)
		if keras.__version__[0] == '1': 
		    return Merge([sent_weights,forward_doc], mode='mul')
		else:
		    return Multiply()([sent_weights, forward_doc])

	def wordpool(self, encoded_words):
		""" Compose a sentence representation given the encoded word
		    vectors in the given word sequence. """
		cur_shape = encoded_words._keras_shape
 		reshape = Reshape((self.args['spad'],self.args['swpad'],cur_shape[2]),
							input_shape=cur_shape)
		if not self.attention_mode:
			return K.mean(reshape(encoded_words),axis=2)
		return K.sum(reshape(encoded_words),axis=2)

	def wordpool_output(self, input_shape):
		""" Defines the dimensions of the resulting sentence vector. """
		return tuple([None, self.args['spad'], input_shape[2]])

	def sentencepool(self, encoded_sentences):
		""" Compose a document representation given the encoded sentence
		 	vectors in the given sentence sequence. """
		if not self.attention_mode:
			return K.mean(encoded_sentences, axis=1)
		return K.sum(encoded_sentences, axis=1)

	def sentencepool_output(self, input_shape):
		""" Defines the dimensions of the resulting document vector. """
		return tuple([None, input_shape[2]])

	def submax(self, x):
		""" Subtracts from each vector the value of its dimension with
		    the maximal value. """
		return x - K.max(x, axis=-1, keepdims=True)

	def submax_output(self, input_shape):
		""" Defines the dimensions of the output of the submax function. """
		return tuple(input_shape)

	def build_model(self, encoders, attention, num_labels):
		""" Builds a hierarchical attention eural network model based
		    on a given set of encoders and attention mechanisms. """
		input_model = Sequential()
		if self.args['enc'] in ["bigru", "attbigru"]:
			words = Input(shape=(self.args['wpad'],self.args['wdim'],))
			forward_words = encoders['sent_enc'](words)
			backward_words = encoders['backsent_enc'](words)
			if keras.__version__[0] == '1':
				bigru_words = Merge([forward_words, backward_words], mode='concat', concat_axis=1)
			else:
				bigru_words = Concatenate()([forward_words, backward_words])
			if self.args['enc'] == "attbigru":
				bigru_words = self.word_attention(bigru_words, attention)
			word_pooling = Lambda(self.wordpool, output_shape=self.wordpool_output)
			sentences = word_pooling(bigru_words)
			forward_sentences = encoders['doc_enc'](sentences)
			backward_sentences = encoders['backdoc_enc'](sentences)
			if keras.__version__[0] == '1':
				bigru_sentences = Merge([forward_sentences, backward_sentences], mode='concat', concat_axis=1)
			else:
				bigru_sentences = Concatenate()([forward_sentences, backward_sentences])
			if self.args['enc'] == "attbigru":
				bigru_sentences = self.sentence_attention(bigru_sentences, attention)
			sentence_pooling = Lambda(self.sentencepool, output_shape=self.sentencepool_output)
			document = sentence_pooling(bigru_sentences)
		elif self.args['enc'] in ["dense", "attdense", "gru", "attgru"]:
			words = Input(shape=(self.args['wpad'],self.args['wdim'],))
			forward_words = encoders['sent_enc'](words)
			if self.args['enc'] in ["attdense","attgru"]:
				forward_words = self.word_attention(forward_words, attention)
			word_pooling = Lambda(self.wordpool, output_shape=self.wordpool_output)
			sentences = word_pooling(forward_words)
			forward_sentences = encoders['doc_enc'](sentences)
			if self.args['enc'] in ["attdense","attgru"]:
				forward_sentences = self.sentence_attention(forward_sentences, attention)
			sentence_pooling = Lambda(self.sentencepool, output_shape=self.sentencepool_output)
			document = sentence_pooling(forward_sentences)
		classifier = Dense(num_labels, input_dim=(document._keras_shape[1]), activation='sigmoid')
		return words, classifier(document)

	def build_multilingual_model(self, labels):
		""" Builds a multilingual hierarchical attention neural network
		    model based on a given component sharing configurations. """
		inputs, outputs = [], []
 		if self.args['share'] == "enc":
			encoders = self.build_encoders()
		elif self.args['share'] == "att":
			attention = self.build_attention(lang='both')
		elif self.args['share'] == "both":
			encoders = self.build_encoders()
			attention = self.build_attention(lang='both')
		for l, language in enumerate(self.args['languages']):
			if self.args['share'] == "enc":
				attention = self.build_attention(lang=language)
			elif self.args['share'] == "att":
				encoders = self.build_encoders()
			elif self.args['share'] == "none":
				encoders = self.build_encoders()
				attention = self.build_attention(lang=language)
			words, preds = self.build_model(encoders, attention, len(labels[l]))
			inputs.append(words)
			outputs.append(preds)
		if self.single_language:
			inputs, outputs = inputs[0], outputs[0]
		input_model = Model(inputs=inputs, outputs=outputs)
		input_model.compile(loss='binary_crossentropy', optimizer="adam", metrics=["accuracy"])
		self.model = input_model
		self.forward_attention()
		return self.model

	def forward_attention(self):
		""" Define functions to get the attention scores at both levels. """
		self.watts, self.satts = [], []
		for l, language in enumerate(self.args['languages']):
			name_watt = "%s_watt" % language
			name_satt = "%s_satt" % language
			if len(self.args['languages']) > 1 and self.args['share'] != 'enc':
				name_watt = "both_watt"
				name_satt = "both_satt"
			if self.model.get_layer(name_watt) is not None:
				outpos = l if name_watt.find('both') > -1 else 0
				watt = theano.function([self.model.layers[l].input],
                       					self.model.get_layer(name_watt).get_output_at(outpos),
                       					allow_input_downcast=True)
				satt = theano.function([self.model.layers[l].input],
                       					self.model.get_layer(name_satt).get_output_at(outpos),
                       					allow_input_downcast=True)
				self.watts.append(watt)
				self.satts.append(satt)

	def fit(self, X_train, Y_train, X_val, Y_val, labels, wvecs, vocabs):
		""" Trains the model using stochastic gradient descent. At each epoch
		    epoch, it stores the parameters of the model and its performance
		    on the validation set. """
		resume_path, resume_epoch = self.find_checkpoint()
		errors, prs, recs, fs = [], [], [], []
		val_scores, train_scores = [], []
		if args['seed'] is not None:
			np.random.seed(self.args['seed'])
		for e in range(self.args['ep']):
			if resume_epoch > 1 and e < resume_epoch:
				continue
			print "\nEpoch %d/%d" % (e+1,self.args['ep'])
			batch, elapsed, curbatch = 0, 0, 0
			all_pred, all_real = [], []
			while( batch < (self.args['ep_size']/self.args['bs']) ):
				X_vecs, Y_vecs, start_time = [], [], time.time()
				for l in range(len(self.args['languages'])):
					idxs = np.random.randint(len(X_train[l]), size=self.args['bs']).tolist()
					cur_x = X_train[l][idxs]; cur_y = Y_train[l][idxs]
					x_vecs, y_vecs = load_vectors(wvecs[l], labels[l], cur_x, cur_y,
											self.args['swpad'], self.args['spad'])
					X_vecs.append(np.array(x_vecs));Y_vecs.append(np.array(y_vecs))
				if self.single_language:
					err = self.model.train_on_batch(X_vecs[0], Y_vecs[0])[0]
					preds = self.model.predict(X_vecs[0],batch_size=self.args['bs'])
				else:
					err = self.model.train_on_batch(X_vecs, Y_vecs)[0]
					preds = self.model.predict(X_vecs,batch_size=self.args['bs'])
				pr, rec, f = self.get_avgresults(preds, Y_vecs)
				errors.append(err); prs.append(pr); recs.append(rec); fs.append(f)
				progress = ((batch+1)*self.args['bs'])*30./(self.args['ep_size'])
				elapsed += time.time() - start_time
				stat_args = ( ("="*(int(progress))).ljust(30,'.'), round(elapsed),
							sum(errors)/len(errors),  sum(prs)/len(prs),
							sum(recs)/len(recs), sum(fs)/len(fs) )
				progress = ("%d/%d"%(((batch+1)*self.args['bs']), self.args['ep_size'])).ljust(15)
				stats = "[%s] - %ds - loss: %.4f - p: %.4f - r: %.4f - f1: %.4f\r" % stat_args
				sys.stdout.write( progress + stats )
				sys.stdout.flush()
				batch += 1; curbatch += self.args['bs']
			if resume_path is not None and e == 0:
				print "\n[*] Loading initial weights from %s" % resume_path
				self.model.load_weights(resume_path)
			train_score = (sum(prs)/len(prs), sum(recs)/len(recs), sum(fs)/len(fs))
			lang_scores = []
			for l in range(len(X_train)):
				print "\n[*] Validating on %s..." % self.args['languages'][l]
				reals, preds = self.eval(l, X_val[l],Y_val[l], wvecs[l], labels[l], L=len(X_train))
				val_score = self.get_results(reals, preds>self.args['t'])
				lang_scores.append(val_score)
			val_scores.append(val_score)
			train_scores.append(train_score)
			for l,language in enumerate(self.args['languages']):
				self.save_model(language, e, train_score, lang_scores[l])
		return train_scores, val_scores

	def find_checkpoint(self):
		""" Check if there is a stored model to resume from. """
		try:
			path = "%s/%s/" % (self.args['path'], self.args['languages'][0])
			fnames = [f for f in os.listdir(path) if f.find('weights') > -1]
			cur_e = np.sort([int(f.split('_')[1].split('-')[0]) for f in fnames])[-1]
			cur_idx = np.argsort([int(f.split('_')[1].split('-')[0]) for f in fnames])[-1]
			resume_path = path + fnames[cur_idx]
			if cur_e > 0:
				self.model.load_weights(resume_path)
			print "[*] Resuming from epoch %d... (%s)" % (cur_e + 1, resume_path)
			return resume_path, cur_e + 1
		except:
			print "[*] No stored model to resume from. "
			return None, 0

	def get_avgresults(self, preds, Y_vecs):
		""" Return the average precision, recall and f-measure computed
			over all languages. """
		pr, rec, f, = [], [], []
		if self.single_language:
			preds = [preds]
		for l, pred in enumerate(preds):
			pred = pred>self.args['t']
			prf = self.get_results(Y_vecs[l],  pred, print_result=False)
			pr.append(prf[0]);rec.append(prf[1]);f.append(prf[2])
		return sum(pr)/len(pr), sum(rec)/len(rec), sum(f)/len(f)

	def save_model(self, language, epoch, train_score, val_score):
		""" Store model and validation score at each epoch. """
		name = "epoch_%d" % epoch
		path = "%s/%s/" % (self.args['path'], language)
		if not os.path.exists(path):
			os.makedirs(path)
		json_string = self.model.to_json()
		open(path+self.args['args_file'], 'w').write(json.dumps(self.args, indent=4))
		self.model.save_weights(path+'%s-weights.h5' % name)
		open(path+'%s-val.txt' % name,'w').write(' '.join([str(v) for v in val_score]))
		open(path+'%s-train.txt' % name,'w').write(' '.join([str(v) for v in train_score]))

	def eval(self, cur_lang, x, y,  wvec, labels, bs=16, av='micro', L=0, source=None):
		""" Evaluate model on the given validation or test set. """
		cur_lang = 0 if source is not None else cur_lang
		preds, real, watts, satts = [], [], [], []
		batch, elapsed, curbatch = 0, 0, 0
		while batch < len(x)/(1.0*bs):
			cur_x = x[curbatch:curbatch+bs]
			cur_y = y[curbatch:curbatch+bs]
			x_vecs, y_vecs = load_vectors(wvec, labels, cur_x, cur_y, self.args['swpad'], self.args['spad'])
			if source is None and not self.single_language:
				pred = self.model.predict([np.array(x_vecs) for i in range(L)])[cur_lang]
			else:
				pred = self.model.predict(np.array(x_vecs))
			if self.args['store_test'] and not self.args['train']:
				watts.append(self.watts[cur_lang](x_vecs))
				satts.append(self.satts[cur_lang](x_vecs))
			preds.append(pred); real.append(y_vecs)
			sys.stdout.write(("\t%d/%d\r"%(((batch+1)*bs), len(x))))
			sys.stdout.flush()
			batch += 1; curbatch += bs
		reals = np.array([rr for r in real for rr in r])
		preds = np.array([pp for p in preds for pp in p])
		if self.args['store_test'] and not self.args['train']:
			watts = np.array([ww for w in watts for ww in w])
			satts = np.array([ss for s in satts for ss in s])
			return reals, preds, watts, satts
 		return reals, preds

	def get_results(self, reals, preds, av="micro", print_result=True):
		""" Calculates and prints precision, recall and f-measure based
		 	on the real and predicted categories. """
		prf = precision_recall_fscore_support(reals, preds, average=av)
		if print_result:
			print "\t**val p: %.5f - r: %.5f - f: %.5f " % (prf[0], prf[1], prf[2])
		return [ prf[0], prf[1], prf[2] ]
