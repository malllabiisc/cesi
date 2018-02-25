from helper import *

from sideInfo  	 import SideInfo 		# For processing data and side information
from embeddings  import Embeddings 		# For learning embeddings
from cluster 	 import Clustering 		# For clustering learned embeddings
from metrics 	 import evaluate 		# Evaluation metrics

reload(sys);
sys.setdefaultencoding('utf-8')			# Swtching from ASCII to UTF-8 encoding

''' *************************************** DATASET PREPROCESSING **************************************** '''

class CESI_Main(object):

	def __init__(self, args):
		self.p = args
		self.logger  = getLogger(args.name, args.log_dir, args.config_dir)
		self.logger.info('Running {}'.format(args.name))
		self.read_triples()

	def read_triples(self):
		self.logger.info('Reading Triples')

		fname = self.p.out_path + self.p.file_triples	# File for storing processed triples
		self.triples_list = []				# List of all triples in the dataset
		self.amb_ent 	  = ddict(int)			# Contains ambiguous entities in the dataset
		self.amb_mentions = {} 				# Contains all ambiguous mentions
		self.isAcronym    = {} 				# Contains all mentions which can be acronyms

		if not checkFile(fname):
			self.ent2wiki = ddict(set)
			with codecs.open(args.data_path, encoding='utf-8', errors='ignore') as f:
				for line in f:
					trp = json.loads(line.strip())

					trp['raw_triple'] = trp['triple']
					sub, rel, obj     = trp['triple']

					if sub.isalpha() and sub.isupper(): self.isAcronym[proc_ent(sub)] = 1		# Check if the subject is an acronym
					if obj.isalpha() and obj.isupper(): self.isAcronym[proc_ent(obj)] = 1		# Check if the object  is an acronym

					sub, rel, obj = proc_ent(sub), trp['triple_norm'][1], proc_ent(obj)		# Get Morphologically normalized subject, relation, object
					if len(sub) == 0  or len(rel) == 0 or len(obj) == 0: continue  			# Ignore incomplete triples

					trp['triple'] 		= [sub, rel, obj]
					trp['triple_unique']	= [sub+'|'+str(trp['_id']), rel, obj+'|'+str(trp['_id'])]
					trp['ent_lnk_sub']	= trp['entity_linking']['subject']
					trp['ent_lnk_obj']	= trp['entity_linking']['object']
					trp['true_sub_link']	= trp['true_link']['subject']
					trp['true_obj_link']	= trp['true_link']['object']
					trp['rel_info']		= trp['kbp_info']					# KBP side info for relation

					self.triples_list.append(trp)

			with open(fname, 'w') as f: 
				f.write('\n'.join([json.dumps(triple) for triple in self.triples_list]))
				self.logger.info('\tCached triples')
		else:
			self.logger.info('\tLoading cached triples')
			with open(fname) as f: 
				self.triples_list = [json.loads(triple) for triple in f.read().split('\n')]

		''' Identifying ambiguous entities '''
		amb_clust = {}
		for trp in self.triples_list:
			sub = trp['triple'][0]
			for tok in sub.split():
				amb_clust[tok] = amb_clust.get(tok, set())
				amb_clust[tok].add(sub)

		for rep, clust in amb_clust.items():
			if rep in clust and len(clust) >= 3:
				self.amb_ent[rep] = len(clust)
				for ele in clust: self.amb_mentions[ele] = 1

		''' Ground truth clustering '''
		self.true_ent2clust = ddict(set)
		for trp in self.triples_list:
			sub_u = trp['triple_unique'][0]
			self.true_ent2clust[sub_u].add(trp['true_sub_link'])
		self.true_clust2ent = invertDic(self.true_ent2clust, 'm2os')


	def get_sideInfo(self):
		self.logger.info('Side Information Acquisition')
		fname = self.p.out_path + self.p.file_sideinfo_pkl

		if not checkFile(fname):
			self.side_info = SideInfo(self.p, self.triples_list, self.amb_mentions, self.amb_ent, self.isAcronym)
			self.logger.info('\tEntity Linking Side info'); 		self.side_info.wikiLinking()					# Entity Linking side information
			self.logger.info('\tPPDB Side info'); 				self.side_info.ppdbLinking()					# PPDB side information
			self.logger.info('\tWord Sense Disamb Side info'); 		self.side_info.wordnetLinking()					# Word-sense disambiguation side information
			self.logger.info('\tMorphological Normalization Side info'); 	self.side_info.morphNorm()					# Morphological normalization side information
			self.logger.info('\tToken Overlap Side info'); 			self.side_info.tokenOverlap(self.amb_mentions, self.amb_ent) 	# IDF Token Overlap side information
			self.logger.info('\tAMIE Side info'); 				self.side_info.amieInfo()					# AMIE side information
			self.logger.info('\tKBP Side info'); 				self.side_info.kbpLinking()					# KBP side information

			del self.side_info.file
			pickle.dump(self.side_info, open(fname, 'wb'))
			self.logger.info('\tCached Side Information')
		else:
			self.logger.info('\tLoading cached Side Information')
			self.side_info = pickle.load(open(fname, 'rb'))


	def embedKG(self):
		self.logger.info("Embedding NP and relation phrases");

		fname1 = self.p.out_path + self.p.file_entEmbed
		fname2 = self.p.out_path + self.p.file_relEmbed

		if not checkFile(fname1) or not checkFile(fname2):
			embed = Embeddings(self.p, self.side_info, self.logger)
			embed.fit()

			self.ent2embed = embed.ent2embed			# Get the learned NP embeddings
			self.rel2embed = embed.rel2embed			# Get the learned RP embeddings

			pickle.dump(self.ent2embed, open(fname1, 'wb'))
			pickle.dump(self.rel2embed, open(fname2, 'wb'))
		else:
			self.logger.info('\tLoading cached Embeddings')
			self.ent2embed = pickle.load(open(fname1, 'rb'))
			self.rel2embed = pickle.load(open(fname2, 'rb'))

	def cluster(self):
		self.logger.info('Clustering NPs and relation phrases');

		fname1 = self.p.out_path + self.p.file_entClust
		fname2 = self.p.out_path + self.p.file_relClust

		if not checkFile(fname1) or not checkFile(fname2):
			
			self.sub2embed, self.sub2id = {}, {}		# Clustering only subjects
			for sub_id, eid in enumerate(self.side_info.isSub.keys()):
				self.sub2id[eid]    	= sub_id
				self.sub2embed[sub_id] = self.ent2embed[eid]
			self.side_info.id2sub = invertDic(self.sub2id)

			clust = Clustering(self.sub2embed, self.rel2embed, self.side_info, self.p)
			self.ent_clust = clust.ent_clust
			self.rel_clust = clust.rel_clust

			dumpCluster(fname1, self.ent_clust, self.side_info.id2ent)
			dumpCluster(fname2, self.rel_clust, self.side_info.id2rel)
		else:
			self.logger.info('\tLoading cached Clustering')
			self.ent_clust = loadCluster(fname1, self.side_info.ent2id)
			self.rel_clust = loadCluster(fname2, self.side_info.rel2id)

	def np_evaluate(self):
		self.logger.info('NP Canonicalizing Evaluation');

		cesi_clust2ent = {}
		for rep, cluster in self.ent_clust.items():
			cesi_clust2ent[rep] = set(cluster)
		cesi_ent2clust = invertDic(cesi_clust2ent, 'm2os')

		cesi_ent2clust_u = {}
		for trp in self.side_info.triples:
			sub_u, sub = trp['triple_unique'][0], trp['triple'][0]
			cesi_ent2clust_u[sub_u] = cesi_ent2clust[self.side_info.ent2id[sub]]
		cesi_clust2ent_u = invertDic(cesi_ent2clust_u, 'm2os')

		eval_results = evaluate(cesi_ent2clust_u, cesi_clust2ent_u, self.true_ent2clust, self.true_clust2ent)

		pprint(eval_results)
		self.logger.info('Macro F1: {}, Micro F1: {}, Pairwise F1: {}'.format(eval_results['macro_f1'], eval_results['micro_f1'], eval_results['pairx_f1']))

		self.logger.info('CESI: #Clusters: %d, #Singletons %d'    % (len(cesi_clust2ent_u), 	len([1 for _, clust in cesi_clust2ent_u.items()    if len(clust) == 1])))
		self.logger.info('Gold: #Clusters: %d, #Singletons %d \n' % (len(self.true_clust2ent),  len([1 for _, clust in self.true_clust2ent.items() if len(clust) == 1])))

		# Dump the final results
		fname = self.p.out_path + self.p.file_results
		with open(fname, 'w') as f: f.write(json.dumps(eval_results))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='CESI: Canonicalizing Open Knowledge Bases using Embeddings and Side Information')
	parser.add_argument('-data', 		dest='dataset', 	default='reverb45k', 			help='Dataset to run CESI on')
	parser.add_argument('-split', 		dest='split', 		default='test', 			help='Dataset split for evaluation')
	parser.add_argument('-data_dir', 	dest='data_dir', 	default='./data', 			help='Data directory')
	parser.add_argument('-out_dir', 	dest='out_dir', 	default='./output', 			help='Directory to store CESI output')
	parser.add_argument('-config_dir', 	dest='config_dir', 	default='./config', 			help='Config directory')
	parser.add_argument('-log_dir', 	dest='log_dir', 	default='./log', 			help='Directory for dumping log files')
	parser.add_argument('-ppdb_url', 	dest='ppdb_url', 	default='http://10.24.28.104:9997/', 	help='Assigned name to the run')
	parser.add_argument('-reset',	 	dest="reset", 		action='store_true', 			help='Clear the cached files (Start a fresh run)')
	parser.add_argument('-name', 		dest='name', 		default=None, 				help='Specify name for restoring previous run')

	# Embedding hyper-parameters
	parser.add_argument('-num_neg_samp', 	dest='num_neg_samp', 	default=10,		type=int,	help='Number of Negative Samples')
	parser.add_argument('-nbatches', 	dest='nbatches', 	default=500,		type=int,	help='Number of batches per epoch')
	parser.add_argument('-max_epochs', 	dest='max_epochs', 	default=10,		type=int,	help='Maximum number of epoch')
	parser.add_argument('-lr', 		dest='lr', 		default=0.001,		type=float,	help='Learning rate')
	parser.add_argument('-lambd', 		dest='lambd', 		default=0,		type=float,	help='Regularization constant for embeddings')	
	parser.add_argument('-lambd_wiki', 	dest='lambd_wiki', 	default=1,		type=float,	help='Entity linking side info constant')
	parser.add_argument('-lambd_wnet', 	dest='lambd_wnet', 	default=0.1,		type=float,	help='Word sense disamb side info constant')
	parser.add_argument('-lambd_ppdb', 	dest='lambd_ppdb', 	default=0.1,		type=float,	help='PPDB side info constant')
	parser.add_argument('-lambd_morph', 	dest='lambd_morph', 	default=0.1,		type=float,	help='Morpho Normalization side info constant')
	parser.add_argument('-lambd_idfTok', 	dest='lambd_idfTok', 	default=0,		type=float,	help='IDF Token side info constant')
	parser.add_argument('-lambd_main_obj', 	dest='lambd_main_obj', 	default=0,		type=float,	help='Structural info constant')
	parser.add_argument('-lambd_amie', 	dest='lambd_amie', 	default=0.001,		type=float,	help='AMIE side info constant')
	parser.add_argument('-lambd_kbp', 	dest='lambd_kbp', 	default=0.001,		type=float,	help='KBP side info constant')
	parser.add_argument('-no-norm', 	dest='normalize', 	action='store_false',			help='Normalize embeddings after every epoch')
	parser.add_argument('-margin', 		dest='margin', 		default=0.01,		type=float, 	help='Margin for pairwise objective')
	parser.add_argument('-embed_dims', 	dest='embed_dims', 	default=300,		type=int,	help='Embedding dimension')
	parser.add_argument('-embed_init', 	dest='embed_init', default='glove', choices=['glove', 'random'],help='Method for Initializing NP and Relation embeddings')
	parser.add_argument('-embed_loc', 	dest='embed_loc',  default='./glove/glove.6B.300d_word2vec.txt',help='Location of embeddings to be loaded')
	parser.add_argument('-trainer', 	dest='trainer',    default='pairwise', choices=['stochastic', 'pairwise'], help='Object function for learning embeddings')

	# Clustering hyper-parameters
	parser.add_argument('-linkage', 	dest='linkage',    default='complete', choices=['complete', 'single', 'avergage'], help='HAC linkage criterion')
	parser.add_argument('-thresh_val', 	dest='thresh_val', 	default=.4239, 		type=float, 	help='Threshold for clustering')
	parser.add_argument('-metric', 		dest='metric', 		default='cosine', 			help='Metric for calculating distance between embeddings')
	parser.add_argument('-num_canopy', 	dest='num_canopy', 	default=1,		type=int,	help='Number of caponies while clustering')
	args = parser.parse_args()

	if args.name == None: args.name = args.dataset + '_' + args.split + '_' + time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")

	args.file_triples	= '/triples.txt'		# Location for caching triples
	args.file_entEmbed 	= '/embed_ent.pkl'		# Location for caching learned embeddings for noun phrases
	args.file_relEmbed	= '/embed_rel.pkl'		# Location for caching learned embeddings for relation phrases 
	args.file_entClust	= '/cluster_ent.txt'		# Location for caching Entity clustering results
	args.file_relClust	= '/cluster_rel.txt'		# Location for caching Relation clustering results
	args.file_sideinfo	= '/side_info.txt'		# Location for caching side information extracted for the KG (for display)
	args.file_sideinfo_pkl	= '/side_info.pkl'		# Location for caching side information extracted for the KG (binary)
	args.file_hyperparams	= '/hyperparams.json'		# Location for loading hyperparameters
	args.file_results	= '/results.json'		# Location for loading hyperparameters

	args.out_path  = args.out_dir  + '/' + args.name 				# Directory for storing output
	args.data_path = args.data_dir + '/' + args.dataset + '/' + args.dataset + '_' + args.split  # Path to the dataset
	if args.reset: os.system('rm -r {}'.format(args.out_path))			# Clear cached files if requeste
	if not os.path.isdir(args.out_path): os.system('mkdir -p ' + args.out_path)	# Create the output directory if doesn't exist

	cesi = CESI_Main(args)	# Loading KG triples
	cesi.get_sideInfo()	# Side Information Acquisition
	cesi.embedKG()		# Learning embedding for Noun and relation phrases
	cesi.cluster()		# Clustering NP and relation phrase embeddings
	cesi.np_evaluate()		# Evaluating the performance over NP canonicalization