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

import sys, json, time, os
import argparse, urllib2
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from util import  load_word_vectors

def clean(text):
	""" Removes special characters from a given text. """
	text = text.replace('\n', ' ')
	text = text.replace('\r', ' ')
	return text.replace('\t', ' ')

def extract_wordids(keywords, lang, vocab):
	""" Extracts the word ids for a given set. """
	y_ids = []
	for keyword in keywords:
		keyword = keyword.strip()
		vecs_ids = []
		for word in keyword.split():
			try:
				idy = vocab[lang].index(word)
				vecs_ids.append(idy)
			except:
				continue
		if len(vecs_ids) > 0:
			y_ids.append(vecs_ids)
	return y_ids

def get_label_counts(y_idxs, lang):
	""" Counts the label occurrences in a given set. """
	h = {}
	for y in y_idxs:
		for yy in y:
    			key = "_".join([str(yyy) for yyy in yy])
    			if key not in h:
				h[key] = 1
    			else:
				h[key] += 1
	return h

def re_index(Y, labels):
	""" Re-indexes the target ids to match the label set """
	for i,y in enumerate(Y):
		label_ids = []
		for j, label in enumerate(y):
			str_label = '_'.join([str(wid) for wid in label])
			label_ids.append(labels.index(str_label))
		Y[i] = label_ids
	return Y

def fetch_data(urls, lang, vocab, ltype):
	""" Fetches and pre-processes the specified URLs given 
	    the provided vocabulary. """
	X, Y, skipped = [], [], 0
	for i, url in enumerate(urls[:]):
		title, teaser, body = "", "", ""
		sys.stdout.write("\t%s (%d/%d)\r" % (lang, i+1, len(urls)) )
		sys.stdout.flush()
		try:
			response = urllib2.urlopen(url)
			soup = BeautifulSoup(response.read(), 'html.parser')
			time.sleep(0.5)
		except:
			print "URL not found: %s" % url
			skipped += 1
			continue
		try:
			title = soup.h1.text.lower()
		except:
			pass
		try:
			teaser =  soup.findAll("p", { "class" : "intro" })[0].text.lower()
		except:
			pass
		try:
			region = soup.findAll("div", { "class" : "longText" })[0]
    			related_stories = region.find('div', {'class':'gallery'})
    			if related_stories is not None:    
				related_stories.decompose() # remove related stories
    			body = region.text.lower()
		except:
			pass
		if title == "" and teaser == "" and body == "":
			skipped += 1
			continue
		labels_specific, labels_general = [], []
		if ltype == "kw":
			try:
				sidepanel = soup.findAll("ul", {"class": "smallList"})[0].findAll("li")
				for li in sidepanel:
					if li.strong.text == "Keywords":
						for kw in li.findAll("a"):
							kw = kw.text.strip().lower()
							labels_specific.append(clean(kw))
			except:
				print "URL is missing kw labels: %s" % url
				continue
		if ltype == "rou":
			try:
				for kw in soup.findAll("div", {"id": "navPath"})[0].findAll("a")[1:]:
					kw = kw.text.strip().lower()
					labels_general.append(clean(kw))
			except:
				print "URL is missing rou labels: %s" % url
				continue
		sentences = [clean(title)]
		sentences += sent_tokenize(clean(teaser).strip())
		sentences += sent_tokenize(clean(body.strip()))
		x, x_ids = [], []
		for sentence in sentences:
			vecs, vecs_ids = [], []
			for word in word_tokenize(sentence):
				try:
					idx = vocab[lang].index(word)
					vecs_ids.append(idx)
				except:
					continue
			if len(vecs_ids) > 0:
				x_ids.append(vecs_ids)
		if ltype == "rou":
			y_ids = extract_wordids(labels_general, lang, vocab)
		elif ltype == "kw":
			y_ids = extract_wordids(labels_specific, lang, vocab)
		X.append(x_ids)
		Y.append(y_ids)
	print "\t%s (%d/%d)" % (lang, len(X), len(urls))
	print "Skipped: %d" % skipped
	return X, Y

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Deutsche-Welle data download toolkit. ')
	parser.add_argument('--lang', help='Language of the news articles.')
	parser.add_argument('--urlpath', help='Path where the URLs are stored.')
	parser.add_argument('--outpath', help='Path where the results should be stored.')
	parser.add_argument('--embpath', default='word_vectors', help='Path of the word vectors in pickle format for each language (e.g. english.pkl, etc).')
	parser.add_argument('--ltype', default="rou", help='Type of the categories: specific (kw) or general (rou).')
	args = parser.parse_args()
	wvec, vocab = load_word_vectors(args.lang, '%s/%s.pkl' % (args.embpath, args.lang) )
	print "[*] Fetching dev data..."
	urls_dev = json.load(open(args.urlpath+'/dev/%s.json' % args.lang))
	X_dev, Y_dev = fetch_data(urls_dev['urls'], args.lang, vocab, args.ltype)
	print "[*] Fetching test data..."
	urls_test = json.load(open(args.urlpath+'/test/%s.json' % args.lang))
	X_test, Y_test = fetch_data(urls_test['urls'], args.lang, vocab, args.ltype)
	print "[*] Fetching training data..."
	urls_train = json.load(open(args.urlpath+'/train/%s.json' % args.lang))
	X, Y = fetch_data(urls_train['urls'], args.lang, vocab, args.ltype)
	yh = get_label_counts(Y+Y_dev+Y_test, args.lang)
	Y = re_index(Y, yh.keys()) 
	Y_dev = re_index(Y_dev, yh.keys())
	Y_test = re_index(Y_test, yh.keys()) 
	if not os.path.exists(args.outpath+'/dev'):
	    os.makedirs(args.outpath+'/dev')
	devfile = open(args.outpath+'/dev/%s.json' % args.lang, 'w')
	print "[*] Storing dev data..."
	json.dump({'X_ids': X_dev, 'Y_ids': Y_dev, 'label_ids': yh.keys()}, devfile)
	if not os.path.exists(args.outpath+'/test'):
	    os.makedirs(args.outpath+'/test')
	testfile = open(args.outpath+'/test/%s.json' % args.lang, 'w')
	print "[*] Storing test data..."
	json.dump({'X_ids': X_test, 'Y_ids': Y_test, 'label_ids': yh.keys()}, testfile)
	if not os.path.exists(args.outpath+'/train'):
		    os.makedirs(args.outpath+'/train')
	print "[*] Storing training data..."
	trainfile = open(args.outpath+'/train/%s.json' % args.lang, 'w')
	json.dump({'X_ids': X, 'Y_ids': Y, 'label_ids': yh.keys()}, trainfile)
	print "[-] Finished."
