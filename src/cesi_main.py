from   joblib import Parallel, delayed  	# For parallizing code
import itertools, pathlib, config, json
import pickle, pdb
import sys, operator

from config import *
from helper import *
from pprint import pprint

from input  	 import Input 			# For processing data and side information
from embeddings  import Embeddings 		# For learning embeddings
from cluster 	 import Clustering 		# For clustering learned embeddings
from metrics 	 import evaluate 		# Evaluation metrics

reload(sys);
sys.setdefaultencoding('utf-8')			# Swtching from ASCII to UTF-8 encoding

config.timer = Timer()

if len(sys.argv) < 3:
	print 	'Please provide required arguments: \n\
		<dataset=[base/ambiguous/reverb45k]> <split=[test/valid]>'
	exit(0)

DATASET    = sys.argv[1]	# [base/ambiguous/reverb45k]
DATA_SPLIT = sys.argv[2]	# [valid/test/full]

config.dpath = '../cache/' + DATASET + '_' + DATA_SPLIT 	# Directory for storing results
if os.path.isdir(config.dpath) == False:			# Create the directory if doesn't exist
	os.system('mkdir -p ' + config.dpath)

config.time_info = dict()

''' *************************************** DATASET PREPROCESSING **************************************** '''
config.timer.start('Dataset Loading & Preprocessing')
fname = config.dpath + config.file_triples 		# File for storing processed triples

amb_ent = ddict(int)
amb_mentions = {} 		# Contains all ambiguous mentions
isAcronym = {} 			# Contains all mentions which can be acronyms

triples_list 	= []

if not checkFile(fname):

	''' Reading Triples '''
	db_triples = []

	ent2wiki = ddict(set)

	with open('../data/' + DATASET + '_' + DATA_SPLIT) as f:
		for line in f:
			trp = json.loads(line.strip())

			trp['raw_triple'] = trp['triple']
			sub, rel, obj     = trp['triple']

			if sub.isalpha() and sub.isupper(): isAcronym[proc_ent(sub)] = 1
			if obj.isalpha() and obj.isupper(): isAcronym[proc_ent(obj)] = 1
			sub, rel, obj = proc_ent(sub), trp['reverb_norm_triple'][1], proc_ent(obj)

			trp['wiki_sub_lnk'] = 	[max(trp['coreNLP_wiki_lnk']['sub_lnk'].iteritems(), key=operator.itemgetter(1))[0]] if trp['coreNLP_wiki_lnk']['sub_lnk'] != None else None

			trp['wiki_obj_lnk'] = 	[max(trp['coreNLP_wiki_lnk']['obj_lnk'].iteritems(), key=operator.itemgetter(1))[0]] if trp['coreNLP_wiki_lnk']['obj_lnk'] != None else None

			if trp['wiki_sub_lnk'] != None: ent2wiki[sub].add(trp['wiki_sub_lnk'][0])
			if trp['wiki_obj_lnk'] != None: ent2wiki[obj].add(trp['wiki_obj_lnk'][0])

			trp['triple'] = [sub, rel, obj]
			db_triples.append(trp)

	''' Identifying ambiguous entities '''
	amb_clust = {}
	for ele in db_triples:
		sub, _, _ = ele['triple']

		for tok in sub.split():
			amb_clust[tok] = amb_clust.get(tok, set())
			amb_clust[tok].add(sub)

	for rep, clust in amb_clust.items():
		if rep in clust and len(clust) >= 3:
			amb_ent[rep] = len(clust)
			for ele in clust: amb_mentions[ele] = 1

	''' Storing dataset in the required format '''
	for trp in db_triples:
		entry = dict()

		sub, rel, obj 		= trp['triple']
		if sub == '' or obj ==  '' or rel == '': continue  				# Ignore incomplete triples
		sub_u, rel_u, obj_u 	= sub+'|'+str(trp['_id']), rel, obj+'|'+str(trp['_id']) # For identifying each entity uniquely

		entry['triple'] 	= [sub, rel, obj]
		entry['triple_u']	= [sub_u, rel_u, obj_u]
		entry['raw_triple']	= trp['raw_triple']
		entry['triple_n']	= trp['reverb_norm_triple']		# Morphological normalized [subject, relation, object]
		entry['_id']		= trp['_id']				# Unique id of each triple
		entry['FACC_lnk']	= trp['FACC_lnk'] 			# Contains ground truth linking
		entry['sentences']	= trp['sentences']			# Source sentences of triple
		entry['wiki_sub_lnk']	= trp['wiki_sub_lnk']			# Entity linking info for subject
		entry['wiki_obj_lnk']	= trp['wiki_obj_lnk']			# Entity linking info for object
		entry['rel_info']	= trp['rel_info']			# KBP side info for relation
		
		triples_list.append(entry)

	with open(fname, 'w') as f: 
		f.write('\n'.join([json.dumps(triple) for triple in triples_list]))
else:
	with open(fname) as f: 
		triples_list = [json.loads(triple) for triple in f.read().split('\n')]


''' Ground truth clustering '''
facc_ent2clust = ddict(set)

for trp in triples_list:
	sub_u, _, _ 	= trp['triple_u']
	facc_ent2clust[sub_u].add(trp['FACC_lnk']['sub_lnk'])
facc_clust2ent = invertDic(facc_ent2clust, 'm2os')

config.facc_ent2clust = facc_ent2clust
config.facc_clust2ent = facc_clust2ent

'''************************************** SIDE INFO ACQUISITION **************************************'''
config.timer.start('Side Information Acquisition')
fname = config.dpath + config.file_sideinfo_pkl

if not checkFile(fname):
	inp = Input(triples_list, amb_mentions, amb_ent, isAcronym)
	pickle.dump(inp, open(fname, 'wb'))
else:
	inp = pickle.load(open(fname, 'rb'))

'''******************************* EMBEDDINGS NP and Relation Phrases *********************************'''
config.timer.start("Embedding NP and relation phrases");

params = json.loads(open('./hyper_params.json').read())[DATASET]
config.embed_dims = params['embed_dims']

fname1 = config.dpath + config.file_entEmbed
fname2 = config.dpath + config.file_relEmbed

if not checkFile(fname1) or not checkFile(fname2):
	Embeddings(inp, params)

	pickle.dump(inp.ent_vector, open(fname1, 'wb'))
	pickle.dump(inp.rel_vector, open(fname2, 'wb'))
else:
	inp.ent_vector = pickle.load(open(fname1, 'rb'))
	inp.rel_vector = pickle.load(open(fname2, 'rb'))

'''***************************************** PERFORM CLUSTERING ****************************************'''
config.timer.start('Clustering');

fn_ent = config.dpath + config.file_entClust
fn_rel = config.dpath + config.file_relClust

cluster_params = {
	'linkage':	'complete',
	'metric':	'cosine',
	'criterion':	'distance',
	'thresh':	'given',
	'thresh_val': 	 params['best_thresh'],
	'break_count':	 20,
	'num_canopy': 	 1,
	'search_steps':  100,
	'upper_limit':   'max' #'25perc'
}


if not checkFile(fn_ent) or not checkFile(fn_rel):
	# Clustering only subjects
	inp.sub_vector = {}
	inp.sub2id = {}
	for sub_id, eid in enumerate(inp.isSub.keys()):
		inp.sub2id[eid]    	= sub_id
		inp.sub_vector[sub_id]  = inp.ent_vector[eid]
	inp.id2sub = invertDic(inp.sub2id)

	Clustering(inp, cluster_params)

	dumpCluster(fn_ent, inp.ent_clust, inp.id2ent)
	dumpCluster(fn_rel, inp.rel_clust, inp.id2rel)
else:
	inp.ent_clust = loadCluster(fn_ent, inp.ent2id)
	inp.rel_clust = loadCluster(fn_rel, inp.rel2id)

''' ************************************** NP EVALUATION ************************************ '''
config.timer.start('Evaluation');

cesi_clust2ent = {}
for rep, cluster in inp.ent_clust.items():
	cesi_clust2ent[rep] = set(cluster)
cesi_ent2clust = invertDic(cesi_clust2ent, 'm2os')

cesi_ent2clust_u = {}
for trp in inp.triples:
	sub_u, sub 	= trp['triple_u'][0], trp['triple'][0]
	cesi_ent2clust_u[sub_u] = cesi_ent2clust[inp.ent2id[sub]]
cesi_clust2ent_u = invertDic(cesi_ent2clust_u, 'm2os')

eval_results = evaluate(cesi_ent2clust_u, cesi_clust2ent_u, facc_ent2clust, facc_clust2ent)

pprint(eval_results)
print eval_results['macro_f1'], eval_results['micro_f1'], eval_results['pairx_f1']

print 'CESI: #Clusters: %d, #Singletons %d'    % (len(cesi_clust2ent_u), len([1 for _, clust in cesi_clust2ent_u.items() if len(clust) == 1])) 
print 'Gold: #Clusters: %d, #Singletons %d \n' 	    % (len(facc_clust2ent),   len([1 for _, clust in facc_clust2ent.items() if len(clust) == 1]))

# Dump the final results
fname = config.dpath + config.file_evalResults
with open(fname, 'w') as f: 
	f.write(json.dumps(eval_results) + '\n')