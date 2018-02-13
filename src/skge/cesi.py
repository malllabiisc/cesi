import numpy as np
from skge.base import Model
from skge.util import *
from skge.param import normless1, Parameter
import skge.actfun as af
import pdb
from memory_profiler import profile


class CESI(Model):

	def __init__(self, *args, **kwargs):
		super(CESI, self).__init__(*args, **kwargs)

		self.add_hyperparam('sz' 	, args[0])			# sz is (N, M, N), N:#Entities, M:#Relations
		self.add_hyperparam('ncomp'	, args[1])			# Dimension of embeddings for entities and relations
		self.add_hyperparam('af'	, kwargs.pop('af', af.Sigmoid))	# Activation function to use
		self.add_hyperparam('labmd'	, kwargs.pop('lambd', 0.0))	# Regularization constant for W
		self.add_hyperparam('lambd_side', kwargs.pop('lambd_side', {})) # Regularization constant for linked entities
		self.add_hyperparam('init'	, kwargs.pop('init', 'nunif'))	# Method for initializing params (default = nusnif, normalized uniform)

		E_init = kwargs.pop('E_init', None)
		R_init = kwargs.pop('R_init', None)
		self.inp = kwargs.pop('inp', None)
		self.add_param('E', (self.sz[0], self.ncomp), post=normless1, value = E_init)	# Passing sz = (N, ncomp): ncomp-dim embedding for each entity
		self.add_param('R', (self.sz[2], self.ncomp), value = R_init)			# Passing sz = (M, ncomp): ncomp-dim embedding for each relation

	# Taking subject(ss), predicate(ps), object[os] ids
	def scores(self, ss, ps, os):						# Computes r.(e_s o e_o)
		return np.sum(self.R[ps] * ccorr(self.E[ss], self.E[os]), axis=1)

	def gradients(self, xys):						# Compute the gradient given list of triples (postive and negative)
		ss, ps, os, ys = unzip_triples(xys, with_ys=True)		# Separates out list of sub, pred, obj, label

		yscores 	= ys * self.scores(ss, ps, os)			# Compute label * r.(e_s o e_o) for all triples | yscr = (y_i * eta_i)
		self.loss 	= np.sum(np.logaddexp(0, -yscores))		# Compute logistic loss 			| sum{ log(1 + exp(-yscr)) } -> Scalar
		#preds = af.Sigmoid.f(yscores)
		fs = -(ys * af.Sigmoid.f(-yscores))[:, np.newaxis]		# np.newaxis: Converts (n,) -> (nx1) 		| -(yi * sig(-yscr)): diff of log(1+exp(-yscr)) wrt eta_i
		#self.loss -= np.sum(np.log(preds))

		ridx, Sm, n = grad_sum_matrix(ps)				# ridx: unique preds, Sm: (len(ridx) x len(ps)) matrix, n: count vector 
		gr = self.lambd_side['main_obj'] * Sm.dot(fs * ccorr(self.E[ss], self.E[os])) / n 		# diff(eta,r)= es o eo: vec (len(fs),) dot with rows of Sm: vec (len(ridx), ) | diff(f,r) = diff(f,eta) * diff(eta, r)
		gr += self.labmd * self.R[ridx]					# Diff from regularization term

		eidx, Sm, n = grad_sum_matrix(list(ss) + list(os))		# eidx: unique entities, Sm: (len(eidx) x len(ss+os)) matrix, n: count vector 
		ge = self.lambd_side['main_obj'] * Sm.dot(np.vstack((
			fs * ccorr(self.R[ps], self.E[os]),			# diff(eta,ss) = ps o os	| diff(f,s) = diff(f,eta) * diff(eta, s)
			fs * cconv(self.E[ss], self.R[ps])			# diff(eta,os) = ps * ss	| diff(f,s) = diff(f,eta) * diff(eta, s)
		))) / n
		
		ge += self.labmd * self.E[eidx]					# Diff from regularization term

		ge, gr = self.sideInfo(ge, gr, eidx, ridx)
		
		return {'E': (ge, eidx), 'R':(gr, ridx)}			# Return the updates


	def sideInfo(self, ge, gr, eidx, ridx):
		''' Entity Side Info '''
		ge = self.updateSideInfo(ge, 	self.E, eidx, self.inp.ent2wiki,  self.inp.id2ent, self.lambd_side['ent_wiki'], 'm2ol')
		ge = self.updateSideInfo(ge, 	self.E, eidx, self.inp.ent2ppdb,  self.inp.id2ent, self.lambd_side['ent_ppdb'], 'm2ol')
		ge = self.updateSideInfo(ge, 	self.E, eidx, self.inp.ent2wnet,  self.inp.id2ent, self.lambd_side['ent_wnet'], 'm2ol')
		ge = self.updateSideInfo(ge, 	self.E, eidx, self.inp.ent2morph, self.inp.id2ent, self.lambd_side['ent_morph'], 'm2ol')
		ge = self.updateSideInfoVal(ge, self.E, eidx, self.inp.ent2idfTok,self.inp.id2ent, self.lambd_side['ent_idfTok']) 

		''' Relation Side Info '''
		gr = self.updateSideInfo(gr, 	self.R, ridx, self.inp.rel2kbp,   self.inp.id2rel, self.lambd_side['rel_kbp'],  'm2ol')
		gr = self.updateSideInfo(gr, 	self.R, ridx, self.inp.rel2ppdb,  self.inp.id2rel, self.lambd_side['rel_ppdb'], 'm2ol')
		gr = self.updateSideInfo(gr, 	self.R, ridx, self.inp.rel2wnet,  self.inp.id2rel, self.lambd_side['rel_wnet'], 'm2ol')
		gr = self.updateSideInfo(gr, 	self.R, ridx, self.inp.rel2amie,  self.inp.id2rel, self.lambd_side['rel_amie'], 'm2o')
		gr = self.updateSideInfo(gr, 	self.R, ridx, self.inp.rel2morph, self.inp.id2rel, self.lambd_side['rel_morph'],'m2ol')
		gr = self.updateSideInfoVal(gr, self.R, ridx, self.inp.rel2idfTok,self.inp.id2rel, self.lambd_side['rel_idfTok']) 

		return ge, gr

	def updateSideInfo(self, grad, embed, id_list, id2clust, id2str, lambd, mode = 'm2o'):
		pairs, Z = getPairs(id_list, id2clust, mode)

		for x1, x2 in pairs:
			str1, str2 = id2str[x1].split('|')[0], id2str[x2].split('|')[0]
			if str1 == str2: continue

			idx1, idx2 = np.where(id_list == x1)[0][0], np.where(id_list == x2)[0][0]	# Get indices in update matrix
			grad[idx1] += (embed[x1] - embed[x2]) * (lambd / Z)
			grad[idx2] -= (embed[x1] - embed[x2]) * (lambd / Z)

		del pairs, Z
		return grad

	def updateSideInfoPair(self, grad, embed, id_list, pair2clust, id2str, lambd):

		ent_affected = dict()
		pair_list = []

		for x1, x2 in itertools.combinations(id_list, 2):
			if (x1, x2) not in pair2clust and (x2, x1) not in pair2clust: continue
			str1, str2 = id2str[x1].split('|')[0], id2str[x2].split('|')[0]
			if str1 == str2: continue
			pair_list.append((x1, x2))

			ent_affected[x1] = 1
			ent_affected[x2] = 1

		Z = len(ent_affected)

		for x1, x2 in pair_list:
			idx1, idx2 = np.where(id_list == x1)[0][0], np.where(id_list == x2)[0][0]

			grad[idx1] += (embed[x1] - embed[x2]) * (lambd / Z)
			grad[idx2] -= (embed[x1] - embed[x2]) * (lambd / Z)

		return grad

	def updateSideInfoVal(self, grad, embed, id_list, id2clust, id2str, lambd):
		for x1, x2 in itertools.combinations(id_list, 2):
			if (x1, x2) not in id2clust and (x2, x1) not in id2clust: continue
			
			str1, str2 = id2str[x1].split('|')[0], id2str[x2].split('|')[0]
			if str1 == str2: continue
			idx1, idx2 = np.where(id_list == x1)[0][0], np.where(id_list == x2)[0][0]

			score = id2clust[(x1,x2)] if (x1,x2) in id2clust else id2clust[(x2,x1)]
			grad[idx1] += (embed[x1] - embed[x2]) * (lambd * score)
			grad[idx2] -= (embed[x1] - embed[x2]) * (lambd * score)

		return grad

	# @profile
	def pairwise_gradients(self, pxs, nxs):
		if len(pxs) == 0 or len(nxs) == 0: return None

		# indices of positive examples
		sp, pp, op = unzip_triples(pxs)					# Separate out sub, pred, obj of positive triples
		# indices of negative examples
		sn, pn, on = unzip_triples(nxs)					# Separate out sub, pred, obj of negative triples

		pscores = self.af.f(self.scores(sp, pp, op))			# Compute sig(eta_{pos})  | eta = r.(es o eo)
		nscores = self.af.f(self.scores(sn, pn, on))			# Compute sig(eta_{neg}) 

		#print("avg = %f/%f, min = %f/%f, max = %f/%f" % (pscores.mean(), nscores.mean(), pscores.min(), nscores.min(), pscores.max(), nscores.max()))

		# find examples that violate margin				# Pairwise ranking loss: sum{ max(0, gamma - sig(eta-) - sig(eta+))}
		ind = np.where(nscores + self.margin > pscores)[0]		# List comes with all pairs already formed | Identify places where (gamma + sig(eta-) - sig(eta+) > 0)
		self.nviolations = len(ind)

		if len(ind) == 0: return 					# No update requred

		# helper vars,
		sp, sn = list(sp[ind]), list(sn[ind])				# Getting violators subs, objs, preds for both +ve and -ve triples
		op, on = list(op[ind]), list(on[ind])
		pp, pn = list(pp[ind]), list(pn[ind])
		gpscores = -self.af.g_given_f(pscores[ind])[:, np.newaxis]	# Diff of activation function | pos comes with -ve sign in loss func therefore -ve sign is added 
		gnscores =  self.af.g_given_f(nscores[ind])[:, np.newaxis]

		# object role gradients
		ridx, Sm, n = grad_sum_matrix(pp + pn)				# Calculating gradients for preds of positive and negative triples together
		grp = gpscores * ccorr(self.E[sp], self.E[op])			# diff(f,r) = diff(f,eta) * diff(eta, r), where diff(eta,r)= es o eo
		grn = gnscores * ccorr(self.E[sn], self.E[on])
		#gr = (Sm.dot(np.vstack((grp, grn))) + self.labmd * self.R[ridx]) / n
		gr = self.lambd_side['main_obj'] * Sm.dot(np.vstack((grp, grn))) / n
		gr += self.labmd * self.R[ridx]				# From regularization

		# filler gradients
		eidx, Sm, n = grad_sum_matrix(sp + sn + op + on)		# Calculating gradients for entities in both positive and negative triples together
		geip = gpscores * ccorr(self.R[pp], self.E[op])
		gein = gnscores * ccorr(self.R[pn], self.E[on])
		gejp = gpscores * cconv(self.E[sp], self.R[pp])
		gejn = gnscores * cconv(self.E[sn], self.R[pn])
		ge = self.lambd_side['main_obj'] * Sm.dot(np.vstack((geip, gein, gejp, gejn))) / n
		ge += self.labmd * self.E[eidx]

		''' Entity Side Info '''
		ge, gr = self.sideInfo(ge, gr, eidx, ridx)

		return {'E': (ge, eidx), 'R':(gr, ridx)}