import numpy as np
from skge.base import Model
from skge.util import grad_sum_matrix, unzip_triples
from skge.param import normalize


class TransE(Model):
	"""
	Translational Embeddings of Knowledge Graphs
	"""

	def __init__(self, *args, **kwargs):
		super(TransE, self).__init__(*args, **kwargs)
		self.add_hyperparam('sz', args[0])				# sz is [N, N, M], N:#Entities, M:#Relations
		self.add_hyperparam('ncomp', args[1])				# Dimension of embeddings of entities and relations
		self.add_hyperparam('l1', kwargs.pop('l1', True))		# Dissimilarity metric: l1 norm (True) or l2 norm (False)
		self.add_param('E', (self.sz[0], self.ncomp), post=normalize)	# Initialize Entity embeddings
		self.add_param('R', (self.sz[2], self.ncomp), post=normalize)	# Initialize Relation embeddings

	def _scores(self, ss, ps, os):						# Gives d(h+l,t)
		score = self.E[ss] + self.R[ps] - self.E[os]			# Compute h + l - t
		if self.l1:
			score = np.abs(score)					# Take l1 norm of (h+l-t) as dissimilarity measure
		else:
			score = score ** 2					# Take l2 norm of (h+l-t) as dissimilarity measure
		return -np.sum(score, axis=1)

	def pairwise_gradients(self, pxs, nxs):
		# indices of positive triples
		sp, pp, op = unzip_triples(pxs)					# Separate out sub, pred, obj of positive triples
		# indices of negative triples
		sn, pn, on = unzip_triples(nxs)					# Separate out sub, pred, obj of negative triples

		pscores = self._scores(sp, pp, op)				# Compute loss for positive triples
		nscores = self._scores(sn, pn, on)				# Compute loss for negative triples
		ind = np.where(nscores + self.margin > pscores)[0]		# Get indexes where violation is happening

		# all examples in batch satify margin criterion
		self.nviolations = len(ind)					# Get the number of violations
		if len(ind) == 0: return					# If no violations -> no change required

		sp, sn = list(sp[ind]), list(sn[ind])				# Getting violators subs, objs, preds for both +ve and -ve triples
		op, on = list(op[ind]), list(on[ind])
		pp, pn = list(pp[ind]), list(pn[ind])

		pg = self.E[sp] + self.R[pp] - self.E[op]			# Compute (h+l-t) for postive triples
		ng = self.E[sn] + self.R[pn] - self.E[on]			# Compute (h+l-t) for negative triples

		if self.l1:							# In case L1 norm is used
			pg = np.sign(pg)					# grad = sign(h+l-t) 	| (h+l-t   > 0)	:  1, else: -1
			ng = -np.sign(ng)					# grad = sign(h'+l-t') 	| (h'+l-t' > 0) : -1, else:  1
		else:
			raise NotImplementedError()

		# entity gradients
		eidx, Sm, n = grad_sum_matrix(sp + op + sn + on)
		ge = Sm.dot(np.vstack((pg, -pg, ng, -ng))) / n

		# relation gradients
		ridx, Sm, n = grad_sum_matrix(pp + pn)
		gr = Sm.dot(np.vstack((pg, ng))) / n
		return {'E': (ge, eidx), 'R': (gr, ridx)}
