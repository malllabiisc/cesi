'''
Performs clustering operation on learned embeddings for both NP and relations
Uses HAC method for clustering.
'''
from helper import *

from joblib import Parallel, delayed
import numpy as np, time, random, pdb, config, itertools

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from metrics import *

class Clustering(object):
	def __init__(self, inp, params):
		self.params = params
		self.inp = inp
		self.num_canopy = self.params['num_canopy']

		raw_ent_clust 	= self.getClusters(inp.sub_vector, 'ENT') 	# Clustering entities
		inp.ent_clust	= self.getEntRep(raw_ent_clust, inp.ent_freq)	# Finding entity cluster representative

		self.params['thresh'] = 'given'

		raw_rel_clust 	= self.getClusters(inp.rel_vector, 'REL')	# Clustering relations
		inp.rel_clust	= self.getRelRep(raw_rel_clust)			# Finding relation cluster representative

	def getClusters(self, embed, identity):

		min_dist, mean_dist, max_dist = 0,0,0
		n, m 	= len(embed), config.embed_dims
		X 	= np.empty((n, m), np.float32)

		for i in range(len(embed)): 
			X[i, :] = embed[i]

		dist 	  = pdist(X, 	  metric = self.params['metric'])
		clust_res = linkage(dist, method = self.params['linkage'])

		max_dist  = np.max(clust_res[:,2])
		min_dist  = np.min(clust_res[:,2])
		mean_dist = np.mean(clust_res[:,2])
		perc25_dist = np.percentile(clust_res[:,2], 25)

		if self.params['thresh'] == 'given':
			best_thresh = self.params['thresh_val']

		else:
			pred_ele2clust = dict()
			best_thresh, max_f1, break_count = 0, -1.0e6, 0
			f1_list = []

			if  	self.params['upper_limit'] == 'max':  upper_limit = max_dist
			elif 	self.params['upper_limit'] == 'mean': upper_limit = mean_dist
			else:					 upper_limit = perc25_dist

			print 'Searching over', min_dist, upper_limit

			clustCnt = 0
			for thresh in np.linspace(min_dist, upper_limit, num=self.params['search_steps']):

				labels = fcluster(clust_res, t = thresh, criterion = self.params['criterion']) - 1
				if max(labels) == 0: break

				f1, score_str, num_clust, num_single = self.getScore(labels)
				if max_f1 <= f1: 
					max_f1, best_thresh = f1, thresh
					break_count = 0
				else:
					break_count += 1
					if break_count > self.params['break_count']: break

				clustCnt += 1
				if clustCnt % 1 == 0: 
					print 'Iteration: %d, f1:%f, max_f1:%f %s, thresh:%f, best_thresh:%f, singleton: (%d/%d)' % \
						(clustCnt, f1, max_f1, score_str, thresh, best_thresh, num_clust, num_single)

			print 'Best Threshold', best_thresh
			config.best_thresh = best_thresh
			
		labels = fcluster(clust_res, t = best_thresh, criterion = self.params['criterion']) - 1
		clusters  = [[] for i in range(max(labels) + 1)]

		for i in range(len(labels)): 
			clusters[labels[i]].append(i)

		return clusters

	def makeClusters(self, labels):
		id2clust = {}
		for i in range(len(labels)):
			id2clust[i] = set([labels[i]])

		pred_ent2clust = {}

		for trp in self.inp.triples:
			sub_u, _, obj_u = trp['triple_u']
			sub, _, obj 	= trp['triple']

			sub_id	= self.inp.sub2id[self.inp.ent2id[sub]]

			pred_ent2clust[sub_u] = id2clust[sub_id]

		return pred_ent2clust


	def getScore(self, labels):
		pred_ent2clust = self.makeClusters(labels)
		pred_clust2ent = invertDic(pred_ent2clust, 'm2os')

		num_clust = len(pred_clust2ent)
		num_single = len([clust for _,clust in pred_clust2ent.items() if len(clust) == 1])

		res = evaluate(pred_ent2clust, pred_clust2ent, config.facc_ent2clust, config.facc_clust2ent)
		score_str = '(' + str(res['macro_f1']) + ' | ' + str(res['micro_f1']) + ' | ' + str(res['pair_f1']) + ')'

		return res['micro_f1'] + res['macro_f1'] + res['pair_f1'], score_str, num_clust, num_single

	def getEntRep(self, clusters, ent2freq):
		final_res = dict()

		for cluster in clusters:
			rep, max_freq = cluster[0], -1

			for ent in cluster:
				if ent2freq[ent] > max_freq:
					max_freq, rep = ent2freq[ent], ent

			rep     = self.inp.id2sub[rep]
			cluster = [self.inp.id2sub[ele] for ele in cluster]

			final_res[rep] = cluster

		return final_res

	def getRelRep(self, clusters):
		embed 	  = self.inp.rel_vector
		final_res = {}

		for cluster in clusters:
			# Find the centroid vector for the elements in cluster
			centroid = np.zeros(config.embed_dims)
			for phr in cluster: centroid += embed[phr]
			centroid = centroid / len(cluster)

			# Find word closest to the centroid
			min_dist = float('inf')
			for rel in cluster:
				dist = np.linalg.norm(centroid - embed[rel])

				if dist < min_dist:
					min_dist = dist
					rep = rel

			final_res[rep] = cluster

		return final_res