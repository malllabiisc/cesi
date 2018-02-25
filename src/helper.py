import os, sys, re, pdb, time, argparse, logging, logging.config
import numpy as np, requests, json, operator, pickle, codecs
from numpy.fft import fft, ifft
from nltk.tokenize import sent_tokenize
import itertools, pathlib
from pprint import pprint

from gensim.utils import lemmatize
from nltk.wsd import lesk
from collections import defaultdict as ddict
from joblib import Parallel, delayed

def mergeList(list_of_list):
	return list(itertools.chain.from_iterable(list_of_list))

def unique(l):
	return list(set(l))

def checkFile(filename):
	return pathlib.Path(filename).is_file()

def invertDic(my_map, struct = 'o2o'):
	inv_map = {}

	if struct == 'o2o':				# Reversing one-to-one dictionary
		for k, v in my_map.items():
			inv_map[v] = k

	elif struct == 'm2o':				# Reversing many-to-one dictionary
		for k, v in my_map.items():
			inv_map[v] = inv_map.get(v, [])
			inv_map[v].append(k)

	elif struct == 'm2ol':				# Reversing many-to-one list dictionary
		for k, v in my_map.items():
			for ele in v:
				inv_map[ele] = inv_map.get(ele, [])
				inv_map[ele].append(k)

	elif struct == 'm2os':
		for k, v in my_map.items():
			for ele in v:
				inv_map[ele] = inv_map.get(ele, set())
				inv_map[ele].add(k)

	return inv_map

def dumpCluster(fname, rep2clust, id2name):
	with open(fname, 'w') as f:
		for rep, clust in rep2clust.items():
			f.write(id2name[rep] + '\n')
			for ele in clust:
				f.write('\t' + id2name[ele] + '\n')

def loadCluster(fname, name2id):
	rep2clust = ddict(list)
	with open(fname) as f:
		for line in f:
			if not line.startswith('\t'): 	rep = name2id[line.strip()]
			else: 			  	rep2clust[rep].append(name2id[line.strip()])

	return rep2clust

# Get embedding of words from gensim word2vec model
def getEmbeddings(model, wrd_list, embed_dims):
	embed_list = []

	for wrd in wrd_list:
		if wrd in model.vocab: 	embed_list.append(model.word_vec(wrd))
		else:			embed_list.append(np.random.rand(embed_dims))

	return np.array(embed_list)

# ****************************** QUERYING PPDB SERVICE ***********************************

''' Returns list of PPDB representatives '''
def queryPPDB(ppdb_url, phr_list):
	try:
		data = {"data": phr_list}
		headers = {'Content-Type' : 'application/json'}
		req = requests.post(ppdb_url + 'ppdbAll', data=json.dumps(data), headers=headers)

		if (req.status_code == 200):
			data = json.loads(req.text)
			return data['data']
		else:
			print("Error! Status code :" + str(req.status_code))

	except Exception as e:
		print("Error in getGlove service!! \n\n", e)

def getPPDBclusters(ppdb_url, phr_list, phr2id):
	ppdb_map = dict()
	raw_phr_list = [phr.split('|')[0] for phr in phr_list]
	rep_list = queryPPDB(ppdb_url, raw_phr_list)

	for i in range(len(phr_list)):
		if rep_list[i] == None: continue        # If no representative for phr then skip

		phrId           = phr2id[phr_list[i]]
		ppdb_map[phrId] = rep_list[i]

	return ppdb_map

def getPPDBclustersRaw(ppdb_url, phr_list):
	ppdb_map = dict()
	raw_phr_list = [phr.split('|')[0] for phr in phr_list]
	rep_list = queryPPDB(ppdb_url, raw_phr_list)

	for i, phr in enumerate(phr_list):
		if rep_list[i] == None: continue        # If no representative for phr then skip
		ppdb_map[phr] = rep_list[i]

	return ppdb_map

# ***************************************** TEXT SPLIT ***********************************************
def proc_ent(ent):
	ent = ent.lower().replace('.', ' ').replace('-', ' ').strip().replace('_',' ').replace('|', ' ').strip()
	ent = ' '.join([tok.split('/')[0] for tok in lemmatize(ent.encode())])
	# ent = ' '.join(list( set(ent.split()) - set(config.stpwords)))
	return ent

def wordnetDisamb(sent, wrd):
	res = lesk(sent, wrd)
	if len(dir(res)) == 92:
		return res.name()
	else:
		return None

def getLogger(name, log_dir, config_dir):
	config_dict = json.load(open(config_dir + '/log_config.json'))

	if os.path.isdir(log_dir) == False: 				# Make log_dir if doesn't exist
		os.system('mkdir {}'.format(log_dir))

	config_dict['handlers']['file_handler']['filename'] = log_dir + '/' + name
	logging.config.dictConfig(config_dict)
	logger = logging.getLogger(name)

	std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
	consoleHandler = logging.StreamHandler(sys.stdout)
	consoleHandler.setFormatter(logging.Formatter(std_out_format))
	logger.addHandler(consoleHandler)

	return logger