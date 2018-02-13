'''
Learns embeddings for NPs and relation phrases
'''

import numpy as np, itertools, pdb
import gensim, time, random, config, gc
from nltk.corpus import stopwords
import numpy as np, pdb, pickle
from helper import *
from scipy.cluster.hierarchy import linkage, fcluster
from helper import *
from joblib import Parallel, delayed
from skge import HolE, StochasticTrainer, PairwiseStochasticTrainer, actfun
from skge.sample import LCWASampler
from skge.cesi import CESI
from glove import *
from sklearn.preprocessing import normalize

class Embeddings(object):

	def __init__(self, inp, params):
		self.params 	= params

		N, M 	= len(inp.ent_list), len(inp.rel_list)
		xs	= inp.trpIds
		ys 	= [1] * len(inp.trpIds)
		sz 	= (N, N, M)

		clean_ent_list = []
		for ent in inp.ent_list: clean_ent_list.append(ent.split('|')[0])


		if self.params['embed_init'] == 'glove':
			fname = config.dpath + config.file_embedInit

			if not checkFile(fname):
				E_init	= getGlove(clean_ent_list, params['glove_embed'])
				R_init  = getGlove(inp.rel_list, params['glove_embed'])

				embeds = {
					'E_init': E_init,
					'R_init': R_init
				}
				pickle.dump(embeds, open(fname, 'wb'))

			else:
				embeds = pickle.load(open(fname, 'rb'))
				E_init = embeds['E_init']
				R_init = embeds['R_init']

		else:
			E_init  = np.random.rand(len(clean_ent_list), config.embed_dims)
			R_init  = np.random.rand(len(inp.rel_list),   config.embed_dims)		

		''' Main Algorithm '''
		lambd_side = {
			'ent_wiki': params['lambd_wiki'],
			'ent_ppdb': params['lambd_ppdb'],
			'ent_wnet': params['lambd_wnet'],
			'ent_morph':params['lambd_morph'],
			'ent_idfTok':params['lambd_idfTok'],
			
			'rel_ppdb': params['lambd_ppdb'],
			'rel_wnet': params['lambd_wnet'],
			'rel_amie': params['lambd_amie'],
			'rel_kbp':  params['lambd_kbp'],
			'rel_morph':params['lambd_morph'],
			'rel_idfTok':params['lambd_idfTok'],

			'main_obj': params['lambd_main_obj']
		}
		
		model 	= CESI( (N, M, N),
				params['embed_dims'],
				lambd 		= params['lambd'],
				lambd_side 	= lambd_side,
				E_init		= E_init,
				R_init		= R_init,
				inp 		= inp
			)

		''' Method for getting negative samples '''
		sampler = LCWASampler(params['num_neg_samp'], [0, 2], xs, sz)

		''' Optimizer '''
		if params['trainer'] == 'stochastic':
			trainer = StochasticTrainer(
					model,						# Model
					nbatches	= params['nbatches'],		# Number of batches
					max_epochs 	= params['max_epochs'],		# Max epochs
					learning_rate   = params['lr'],			# Learning rate
					af 		= actfun.Sigmoid,		# Activation function
					samplef 	= sampler.sample,		# Sampling method
					post_epoch	= [self.epoch_callback]		# Callback after each epoch
				)

		else:   trainer = PairwiseStochasticTrainer(
				model,						# Model
				nbatches	= params['nbatches'],		# Number of batches
				max_epochs 	= params['max_epochs'],		# Max epochs
				learning_rate   = params['lr'],			# Learning rate
				af 		= actfun.Sigmoid,		# Activation function
				samplef 	= sampler.sample,		# Sampling method
				margin		= params['margin'],		# Margin
				post_epoch	= [self.epoch_callback]		# Callback after each epoch
			)
		
		trainer.fit(xs, ys)

		for id in inp.id2ent.keys(): inp.ent_vector[id] = trainer.model.E[id]
		for id in inp.id2rel.keys(): inp.rel_vector[id] = trainer.model.R[id]

	def epoch_callback(self, m, with_eval=False):
		if m.epoch % 1 == 0: 
			print 'Epochs: ', m.epoch

		if self.params['normalize']: 			# Normalize embeddings after every epoch
			normalize(m.model.E, copy=False)
			normalize(m.model.R, copy=False)