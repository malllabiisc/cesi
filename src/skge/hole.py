import numpy as np
from skge.base import Model
from skge.util import grad_sum_matrix, unzip_triples, ccorr, cconv
from skge.param import normless1, Parameter
import skge.actfun as af
import pdb

class HolE(Model):

	def __init__(self, *args, **kwargs):
		super(HolE, self).__init__(*args, **kwargs)
		self.add_hyperparam('sz'		, args[0])			# sz is [N, N, M], N:#Entities, M:#Relations
		self.add_hyperparam('ncomp'		, args[1])			# Dimension of embeddings for entities and relations
		self.add_hyperparam('rparam'		, kwargs.pop('rparam', 0.0))	# Regularization for W
		self.add_hyperparam('af'		, kwargs.pop('af', af.Sigmoid))	# Activation function

		E_init = kwargs.pop('E_init', None)
		R_init = kwargs.pop('R_init', None)
		self.add_param('E', (self.sz[0], self.ncomp), post=normless1, value = E_init)	# Passing sz = (N, ncomp): ncomp-dim embedding for each entity
		self.add_param('R', (self.sz[2], self.ncomp), value = R_init)			# Passing sz = (M, ncomp): ncomp-dim embedding for each relation

	# Taking subject(ss), predicate(ps), object[os] ids
	def _scores(self, ss, ps, os):						# Computes r.(e_s o e_o)
		return np.sum(self.R[ps] * ccorr(self.E[ss], self.E[os]), axis=1)

	def gradients(self, xys):						# Compute the gradient given list of triples (postive and negative)
		ss, ps, os, ys = unzip_triples(xys, with_ys=True)		# Separates out list of sub, pred, obj, label

		yscores = ys * self._scores(ss, ps, os)				# Compute label * r.(e_s o e_o) for all triples | yscr = (y_i * eta_i)
		self.loss = np.sum(np.logaddexp(0, -yscores))			# Compute logistic loss 			| sum{ log(1 + exp(-yscr)) } -> Scalar
		#preds = af.Sigmoid.f(yscores)
		fs = -(ys * af.Sigmoid.f(-yscores))[:, np.newaxis]		# np.newaxis: Converts (n,) -> (nx1) 		| -(yi * sig(-yscr)): diff of log(1+exp(-yscr))
		#self.loss -= np.sum(np.log(preds))

		ridx, Sm, n = grad_sum_matrix(ps)				# ridx: unique preds, Sm: (len(ridx) x len(ps)) matrix, n: count vector 
		gr = Sm.dot(fs * ccorr(self.E[ss], self.E[os])) / n 		# diff(eta,r)= es o eo: vec (len(fs),) dot with rows of Sm: vec (len(ridx), ) | diff(f,r) = diff(f,eta) * diff(eta, r)
		gr += self.rparam * self.R[ridx]				# Diff from regularization term

		eidx, Sm, n = grad_sum_matrix(list(ss) + list(os))		# eidx: unique entities, Sm: (len(eidx) x len(ss+os)) matrix, n: count vector 
		ge = Sm.dot(np.vstack((
			fs * ccorr(self.R[ps], self.E[os]),			# diff(eta,ss) = ps o os	| diff(f,s) = diff(f,eta) * diff(eta, s)
			fs * cconv(self.E[ss], self.R[ps])			# diff(eta,os) = ps * ss	| diff(f,s) = diff(f,eta) * diff(eta, s)
		))) / n
		
		ge += self.rparam * self.E[eidx]				# Diff from regularization term

		return {'E': (ge, eidx), 'R':(gr, ridx)}			# Return the updates

	def pairwise_gradients(self, pxs, nxs):
		# indices of positive examples
		sp, pp, op = unzip_triples(pxs)					# Separate out sub, pred, obj of positive triples
		# indices of negative examples
		sn, pn, on = unzip_triples(nxs)					# Separate out sub, pred, obj of negative triples

		pscores = self.af.f(self._scores(sp, pp, op))			# Compute sig(eta_{pos})  | eta = r.(es o eo)
		nscores = self.af.f(self._scores(sn, pn, on))			# Compute sig(eta_{neg}) 

		#print("avg = %f/%f, min = %f/%f, max = %f/%f" % (pscores.mean(), nscores.mean(), pscores.min(), nscores.min(), pscores.max(), nscores.max()))

		# find examples that violate margin				# Pairwise ranking loss: sum{ max(0, gamma - sig(eta-) - sig(eta+))}
		ind = np.where(nscores + self.margin > pscores)[0]		# List comes with all pairs already formed | Identify places where (gamma + sig(eta-) - sig(eta+) > 0)
		self.nviolations = len(ind)

		if len(ind) == 0: return 					# No update requred

		# aux vars,
		sp, sn = list(sp[ind]), list(sn[ind])				# Getting violators subs, objs, preds for both +ve and -ve triples
		op, on = list(op[ind]), list(on[ind])
		pp, pn = list(pp[ind]), list(pn[ind])
		gpscores = -self.af.g_given_f(pscores[ind])[:, np.newaxis]	# Diff of activation function | pos comes with -ve sign in loss func therefore -ve sign is added 
		gnscores =  self.af.g_given_f(nscores[ind])[:, np.newaxis]

		# object role gradients
		ridx, Sm, n = grad_sum_matrix(pp + pn)				# Calculating gradients for preds of positive and negative triples together
		grp = gpscores * ccorr(self.E[sp], self.E[op])			# diff(f,r) = diff(f,eta) * diff(eta, r), where diff(eta,r)= es o eo
		grn = gnscores * ccorr(self.E[sn], self.E[on])
		#gr = (Sm.dot(np.vstack((grp, grn))) + self.rparam * self.R[ridx]) / n
		gr = Sm.dot(np.vstack((grp, grn))) / n
		gr += self.rparam * self.R[ridx]				# From regularization

		# filler gradients
		eidx, Sm, n = grad_sum_matrix(sp + sn + op + on)		# Calculating gradients for entities in both positive and negative triples together
		geip = gpscores * ccorr(self.R[pp], self.E[op])
		gein = gnscores * ccorr(self.R[pn], self.E[on])
		gejp = gpscores * cconv(self.E[sp], self.R[pp])
		gejn = gnscores * cconv(self.E[sn], self.R[pn])
		ge = Sm.dot(np.vstack((geip, gein, gejp, gejn))) / n
		#ge += self.rparam * self.E[eidx]

		return {'E': (ge, eidx), 'R':(gr, ridx)}