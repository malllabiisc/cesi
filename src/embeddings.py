import numpy as np, itertools, pdb
import gensim, time, random, gc
from nltk.corpus import stopwords
import numpy as np, pdb, pickle
from helper import *
from scipy.cluster.hierarchy import linkage, fcluster
from helper import *
from joblib import Parallel, delayed
from skge import HolE, StochasticTrainer, PairwiseStochasticTrainer, actfun
from skge.sample import LCWASampler
from skge.cesi import CESI
from sklearn.preprocessing import normalize

class Embeddings(object):
	"""
	Learns embeddings for NPs and relation phrases
	"""

	def __init__(self, params, side_info, logger):
		self.p 	    = params
		self.logger = logger

		self.side_info  = side_info
		self.ent2embed = {}			# Stores final embeddings learned for noun phrases
		self.rel2embed = {}			# Stores final embeddings learned for relation phrases

	def fit(self):
		N, M 	= len(self.side_info.ent_list), len(self.side_info.rel_list)
		xs	= self.side_info.trpIds
		ys 	= [1] * len(self.side_info.trpIds)
		sz 	= (N, N, M)

		clean_ent_list = []
		for ent in self.side_info.ent_list: clean_ent_list.append(ent.split('|')[0])

		''' Intialize embeddings '''
		if self.p.embed_init == 'glove':
			model = gensim.models.KeyedVectors.load_word2vec_format(self.p.embed_loc, binary=False)
			E_init	= getEmbeddings(model, clean_ent_list, self.p.embed_dims)
			R_init  = getEmbeddings(model, self.side_info.rel_list,   self.p.embed_dims)
		else:
			E_init  = np.random.rand(len(clean_ent_list), self.p.embed_dims)
			R_init  = np.random.rand(len(self.side_info.rel_list),   self.p.embed_dims)

		''' Main Algorithm '''
		lambd_side = {
			'ent_wiki': 	self.p.lambd_wiki,
			'ent_ppdb': 	self.p.lambd_ppdb,
			'ent_wnet': 	self.p.lambd_wnet,
			'ent_morph':	self.p.lambd_morph,
			'ent_idfTok':	self.p.lambd_idfTok,
			
			'rel_ppdb': 	self.p.lambd_ppdb,
			'rel_wnet': 	self.p.lambd_wnet,
			'rel_amie': 	self.p.lambd_amie,
			'rel_kbp':  	self.p.lambd_kbp,
			'rel_morph':	self.p.lambd_morph,
			'rel_idfTok':	self.p.lambd_idfTok,

			'main_obj': 	self.p.lambd_main_obj
		}
		
		model 	= CESI( (N, M, N),
				self.p.embed_dims,
				lambd 		= self.p.lambd,
				lambd_side 	= lambd_side,
				E_init		= E_init,
				R_init		= R_init,
				inp 		= self.side_info
			)

		''' Method for getting negative samples '''
		sampler = LCWASampler(self.p.num_neg_samp, [0, 2], xs, sz)

		''' Optimizer '''
		if self.p.trainer == 'stochastic':
			self.trainer = StochasticTrainer(
					model,						# Model
					nbatches	= self.p.nbatches,		# Number of batches
					max_epochs 	= self.p.max_epochs,		# Max epochs
					learning_rate   = self.p.lr,			# Learning rate
					af 		= actfun.Sigmoid,		# Activation function
					samplef 	= sampler.sample,		# Sampling method
					post_epoch	= [self.epoch_callback]		# Callback after each epoch
				)

		else:   self.trainer = PairwiseStochasticTrainer(
				model,						# Model
				nbatches	= self.p.nbatches,		# Number of batches
				max_epochs 	= self.p.max_epochs,		# Max epochs
				learning_rate   = self.p.lr,			# Learning rate
				af 		= actfun.Sigmoid,		# Activation function
				samplef 	= sampler.sample,		# Sampling method
				margin		= self.p.margin,		# Margin
				post_epoch	= [self.epoch_callback]		# Callback after each epoch
			)
	
		self.trainer.fit(xs, ys)

		for id in self.side_info.id2ent.keys(): self.ent2embed[id] = self.trainer.model.E[id]
		for id in self.side_info.id2rel.keys(): self.rel2embed[id] = self.trainer.model.R[id]

	def epoch_callback(self, m, with_eval=False):
		if m.epoch % 1 == 0: 
			self.logger.info('\tEpochs: {}'.format(m.epoch))

		if self.p.normalize: 						# Normalize embeddings after every epoch
			normalize(m.model.E, copy=False)
			normalize(m.model.R, copy=False)