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
import argparse, json, keras
from models import MHAN
if keras.__version__[0] == "1":
    from keras.utils.visualize_util import plot as plot_model
else:
    from keras.utils import plot_model
from util import load_data, load_word_vectors, pick_best, export

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Multilingual Hierachical Attention Networks toolkit.')
	parser.add_argument('--wdim', type=int, default=40, help='Number of dimensions of the word embeddings. ')
	parser.add_argument('--swpad', type=int, default=30, help='Maximum number of words in a sentence.')
	parser.add_argument('--spad', type=int, default=30, help='Maximum number of sentences in a document.')
	parser.add_argument('--sdim', type=int, default=100, help='Number of hidden dimensions of the word-level encoder.')
	parser.add_argument('--ddim', type=int, default=100, help='Number of hidden dimensions of the sentence-level encoder.')
	parser.add_argument('--ep', type=int, default=200, help='Maximum number of epochs for training.')
	parser.add_argument('--ep_size', type=int, default=25000, help='Maximum number of samples per epoch during training.')
	parser.add_argument('--bs', type=int, default=16, help='Size of batch to be used for training.')
	parser.add_argument('--enc', type=str, default='attdense', help='Type of encoder and pooling layer to be used at each level of the hierarchical model, namely dense, gru or bigru (transformation + average pooling) or attdense, attgru, attbigru (transformation + attention mechanism).')
	parser.add_argument('--act', type=str, default='relu', help='Activation to be used for the Dense layers.')
	parser.add_argument('--gruact', type=str, default='tanh', help='Activation to be used for the GRU/biGRU layers.')
	parser.add_argument('--share', type=str, default='none', help='Component to share in the multilingual model: encoders (enc), attention (att) or both (both).')
	parser.add_argument('--t', type=float, default=0.20, help='Decision threshold to be used for validation or testing.')
	parser.add_argument('--seed', type=int, default=1234, help='Random seed number.')
	parser.add_argument('--args_file', type=str,  default='args.json', help='Name of the file in the experiment folder where the settings of the model are stored.')
	parser.add_argument('--wordemb_path', type=str,  default='word_vectors/', help='<Required:train|test|store_test> Path of the word vectors in pickle format for each language (e.g. english.pkl, etc).')
	parser.add_argument('--languages', nargs='+', default=None, help='<Required:train> Languages to be used for multilingual training.')
	parser.add_argument('--data_path', type=str,  default='data/dw_general', help='<Required:train|test|store_test> Path of the train/, dev/ and test/ folders which contain data in json format for each language.')
	parser.add_argument('--path', type=str,  default='exp/english', help='<Required:train> Path of the experiment in which the model parameters and validation scores are stored after each epoch.')
	parser.add_argument('--target', type=str,  default=None, help='<Required:test> Language in which the testing should be performed.')
	parser.add_argument('--source', type=str,  default=None, help='<Optional:test> Language in which model should be loaded from. Useful only for cross-lingual tagging (user with --store_test option).')
	parser.add_argument('--store_file', type=str, default='results.json', help='<Optional:store_test> Name of the file in the experiment folder where the predictions and attention scores of the model are stored.')
	parser.add_argument('--max_num', type=int, default=500, help='<Optional:store_test> Maximum number of test examples to consider.')
	parser.add_argument('--train', action='store_true', help='Train the model from scratch.')
	parser.add_argument('--test', action='store_true', help='Test the model.')
	parser.add_argument('--store_test', action='store_true', help='Store the predictions and attention scores of the model on the test set.')
	parsed_args = parser.parse_args()
	args = parsed_args.__dict__
	X_ids, Y_ids, XV_ids, YV_ids, XT_ids = [], [], [], [], []
	YT_ids, wvecs, labels, vocabs = [], [], [], []
	if not args['train']:
		to_load = parsed_args.target
		if parsed_args.source:
			to_load = parsed_args.source
		json_string = open("%s/%s/%s" % ( args['path'], to_load, args['args_file']) ).read()
		args = json.loads(json_string)
		if parsed_args.languages is not None:
			args['languages'] = parsed_args.languages
		else:
			parsed_args.languages = args['languages']
		args['train'] = False
		args['path'] = parsed_args.path
		args['source'] = parsed_args.source
		args['target'] = parsed_args.target
		args['test'] = parsed_args.test
		args['store_test'] = parsed_args.store_test
		args['t'] = parsed_args.t
	for language in args['languages']:
		wordemb_path = args['wordemb_path']+'%s.pkl' %  language
		wvec, vocab = load_word_vectors(language, wordemb_path)
		if parsed_args.train:
			train_path = args['data_path']+'/train/%s.json' %  language
			dev_path = args['data_path']+'/dev/%s.json' %  language
			x_ids, y_ids, cur_labels = load_data(path=train_path)
			xv_ids, yv_ids, cur_labels = load_data( path=dev_path)
			print "\tX_train (80%)"+": %d" % len(x_ids)
			print "\tX_val (10%)"+": %d" % len(xv_ids)
			X_ids.append(np.array(x_ids));Y_ids.append(np.array(y_ids))
			XV_ids.append(np.array(xv_ids));YV_ids.append(np.array(yv_ids))
		elif parsed_args.test or parsed_args.store_test:
			test_path = args['data_path']+'/test/%s.json' %  language
			xt_ids, yt_ids, cur_labels = load_data( path=test_path)
		 	print "\tX_test (10%)"+": %d" % len(xt_ids)
		 	if parsed_args.store_test:
		 		max_num = parsed_args.max_num
		 		XT_ids.append(np.array(xt_ids)[:max_num]);YT_ids.append(np.array(yt_ids)[:max_num])
		 	else:
				XT_ids.append(np.array(xt_ids));YT_ids.append(np.array(yt_ids))
		print "\t|V|: %d, |Y|: %d" % (len(vocab[language]),len(cur_labels))
		labels.append(cur_labels)
		wvecs.append(np.array(wvec[language]))
		vocabs.append(vocab[language])
	mhan = MHAN(args)
	mhan.build_multilingual_model(labels)
	if parsed_args.train:
		print "[*] Training model..."
		mhan.fit(X_ids, Y_ids, XV_ids, YV_ids, labels, wvecs, vocabs)
	if parsed_args.test or parsed_args.store_test:
		lang_idx = parsed_args.languages.index(args['target'])
		dev_path = "%s/%s/" % ( args['path'], args['target'])
		source_idx = lang_idx
		if parsed_args.source is not None:
			print "[*] Cross-lingual mode: ON"
			print "[*] Source language: %s" % args['source']
			dev_path = "%s/%s/" % ( args['path'], args['source'])
			source_idx = parsed_args.languages.index(args['source'])
		epoch_num, best_weights_file = pick_best(dev_path)
		mhan.model.load_weights(best_weights_file)
		plot_model(mhan.model, to_file="%sarch.png" % dev_path)
		if parsed_args.store_test:
			print "[*] Storing predictions on %s test..." % args["target"]
			reals, epreds, watts, satts = mhan.eval(lang_idx, XT_ids[lang_idx], YT_ids[lang_idx], wvecs[lang_idx], labels[lang_idx], L=len(parsed_args.languages), source=parsed_args.source)
			out = export(args["target"], lang_idx, source_idx, epreds, watts, satts, XT_ids, YT_ids, vocabs, labels)
			json.dump(out, open("%s%s" % (dev_path,parsed_args.store_file), 'w'))
		else:
			print "[*] Testing model on %s..." % args['target']
			reals, epreds = mhan.eval(lang_idx, XT_ids[lang_idx], YT_ids[lang_idx], wvecs[lang_idx], labels[lang_idx], L=len(args['languages']))
			print "\tthreshold > %.2f" % args['t']
			res = mhan.get_results(reals, epreds>args['t'])
	 		out_file = "%sepoch_%d-test-%.2f.txt" % (dev_path, epoch_num, args['t'])
			open(out_file, 'w').write(' '.join([str(v) for v in res]))
	print "[-] Finished."
