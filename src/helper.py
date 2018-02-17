import time, pathlib, os, config, re, pdb
import numpy as np, requests, json, operator
from numpy.fft import fft, ifft
from nltk.tokenize import sent_tokenize
import config, itertools, pathlib, requests
from gensim.utils import lemmatize
from nltk.wsd import lesk
from collections import defaultdict as ddict

class Timer(object):

	def __init__(self):
		self.msg = ""

	def start(self, msg):
		if self.msg != "": self.stop()
		self.msg = msg
		self.tstart = time.time()

	def stop(self):
		t = time.time() - self.tstart
		if config.time_info != None:
			config.time_info[self.msg] = round(t, 2) + 0.01
		print ( '{} => {} seconds'.format(self.msg, t))
		self.msg = ""

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



# ****************************** QUERYING PPDB SERVICE ***********************************

''' Returns list of PPDB representatives '''
def queryPPDB(phr_list):
	try:
		data = {"data": phr_list}
		headers = {'Content-Type' : 'application/json'}
		req = requests.post(config.ppdb_url + 'ppdbAll', data=json.dumps(data), headers=headers)

		if (req.status_code == 200):
			data = json.loads(req.text)
			return data['data']
		else:
			print ("Error! Status code :" + str(req.status_code))

	except Exception as e:
		print ("Error in getGlove service!! \n\n", e)

def getPPDBclusters(phr_list, phr2id):
	ppdb_map = dict()
	raw_phr_list = [phr.split('|')[0] for phr in phr_list]
	rep_list = queryPPDB(raw_phr_list)

	for i in range(len(phr_list)):
		if rep_list[i] == None: continue        # If no representative for phr then skip

		phrId           = phr2id[phr_list[i]]
		ppdb_map[phrId] = rep_list[i]

	return ppdb_map

def getPPDBclustersRaw(phr_list):
	ppdb_map = dict()
	raw_phr_list = [phr.split('|')[0] for phr in phr_list]
	rep_list = queryPPDB(raw_phr_list)

	for i, phr in enumerate(phr_list):
		if rep_list[i] == None: continue        # If no representative for phr then skip
		ppdb_map[phr] = rep_list[i]

	return ppdb_map


# ******************************* Unicode --> ASCII ***********************************
import unicodedata

def asciize(text):
	try:
		if not isinstance(text, unicode):
			text=text.decode('utf-8')
	except:
		text = text.decode(text, 'latin-1')
		text = text.encode(text, 'utf-8')
		return asciize(text)

	text = unicodedata.normalize('NFKD', text).encode('ascii','ignore')
	return text

def uni2asc(text):
	text = unicodedata.normalize('NFKD', text.decode('utf-8')).encode('ascii','ignore')
	text = text.encode('utf-8')
		
	for k, v in config.html_unicode.items():
		text = text.replace(k, v)

	text = text.replace('\xc2', ' ').replace('\xa0', ' ')
	# text = asciize(text)
	return text

# ***************************************** TEXT SPLIT ***********************************************
def partition(lst, n):
        division = len(lst) / float(n)
        return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in xrange(n) ]

def getChunks(inp_list, chunk_size):
	return [inp_list[x:x+chunk_size] for x in xrange(0, len(inp_list), chunk_size)]

def proc_url(url):
	url = url.lower()
	url = url.replace('http://', '')
	url = url.replace('www.', '')
	url = url.split('/')[0].strip('.')
	return url

def proc_cat(cat):
	cat = cat.lower()
	cat = '/'.join(cat.split('/')[:2])
	cat = cat.replace('|top', '')
	return cat

def proc_ent(ent):
	ent = ent.lower().replace('.', ' ').replace('-', ' ').strip().replace('_',' ').replace('|', ' ').strip()
	ent = ' '.join([tok.split('/')[0] for tok in lemmatize(ent)])
	# ent = ' '.join(list( set(ent.split()) - set(config.stpwords)))
	return ent

def getUrls(url_list, url2topic):
	for url in url_list:
		dom = proc_url(url)
		if dom in url2topic:
			return dom
	return ''

def getMaxKey(my_map):
	if len(my_map) == 0: return ''
	else:
		return max(my_map.items(), key=operator.itemgetter(1))[0]

def len_key(tp):
	return len(tp[1])

def sortPrint(clust2ent, singles=True):
	temp = clust2ent.items()
	temp.sort(reverse = True, key = len_key)

	for rep, clust in temp:
		if not singles and len(clust) == 1: continue
		print rep
		for ele in clust:
			print '\t', ele

def wordnetDisamb(sent, wrd):
	res = lesk(sent, wrd)
	if len(dir(res)) == 92:
		return res.name()
	else:
		return None

def containsNot(a):
	if 'not' in a.split() or 'no' in a.split():
		return True
	else:
		return False

def get_logger(name):
	config_dict = json.load(open('./log_config.json'))
	if os.path.isdir(config.dpath) == False: 
		
	config_dict['handlers']['file_handler']['filename'] = './log/' + name.replace('/', '-')
	logging.config.dictConfig(config_dict)
	logger = logging.getLogger(name)

	std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
	consoleHandler = logging.StreamHandler(sys.stdout)
	consoleHandler.setFormatter(logging.Formatter(std_out_format))
	logger.addHandler(consoleHandler)

	return logger