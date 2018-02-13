"""
Sampling strategies to generate negative examples from knowledge graphs
with an open-world assumption
"""

from copy import deepcopy
from collections import defaultdict as ddict 	# if a key is not found in the dictionary, then instead of a KeyError being thrown, a new entry is created based on type defined
from numpy.random import randint
# from memory_profiler import profile
import sys

class Sampler(object):

	def __init__(self, n, modes, ntries=100):
		self.n = n 			# Number of negative examples
		self.modes = modes 		# Can be [0,1] / [0,1,2] -> Decides what to replace (sub/obj/pred) for getting neg samples
		self.ntries = ntries		# Max number of retries for generating one negative triple

	def sample(self, xys):
		res = []
		for x, _ in xys: 				# For each true triple
			for _ in range(self.n):			# Create n negative samples
				for mode in self.modes:		# mode -> sub(0)/rel(1)/obj(2)
					t = self._sample(x, mode)
					if t is not None:
						res.append(t)
		return res


class RandomModeSampler(Sampler):
	"""
	Sample negative triples randomly
	"""

	def __init__(self, n, modes, xs, sz):
		super(RandomModeSampler, self).__init__(n, modes)
		self.xs = set(xs)
		self.sz = sz

	def _sample(self, x, mode):
		nex = list(x)
		res = None
		for _ in range(self.ntries):
			nex[mode] = randint(self.sz[mode])	# Based on mode randomly select sub/obj/rel
			if tuple(nex) not in self.xs:		# Checking if produced triple is not a part of true set
				res = (tuple(nex), -1.0)	# Got a negative sample
				break
		return res


class RandomSampler(Sampler):

	def __init__(self, n, xs, sz):
		super(RandomSampler, self).__init__(n)
		self.xs = set(xs)
		self.sz = sz

	def _sample(self, x, mode):
		res = None
		for _ in range(self.ntries):
			nex = (randint(self.sz[0]), randint(self.sz[0]), randint(self.sz[1]))	# Generates a completely random triple
			if nex not in self.xs:			# Checks if it is already there in the training set
				res = (nex, -1.0)		# Got a negative sample
				break
		return res


# Couldn't understand this much
class CorruptedSampler(Sampler):

	def __init__(self, n, xs, type_index):
		super(CorruptedSampler, self).__init__(n)
		self.xs = set(xs)
		self.type_index = type_index

	def _sample(self, x, mode):
		nex = list(deepcopy(x))
		res = None
		for _ in range(self.ntries):
			if mode == 2:
				nex[2] = randint(len(self.type_index))
			else:
				k = x[2]
				n = len(self.type_index[k][mode])
				nex[mode] = self.type_index[k][mode][randint(n)]
			if tuple(nex) not in self.xs:
				res = (tuple(nex), -1.0)
				break
		return res


'''Sample negative examples according to the local closed world assumption'''
class LCWASampler(RandomModeSampler):

	def __init__(self, n, modes, xs, sz):
		super(LCWASampler, self).__init__(n, modes, xs, sz)
		self.counts = dict()		# Creating a defaultDictionary
		for s, o, p in xs:			# Counting occurence of (s,p) in the training dataset
			self.counts[(s,p)] = self.counts.get((s,p), 0)
			self.counts[(s, p)] += 1

	def _sample(self, x, mode):
		nex = list(deepcopy(x))
		res = None

		for _ in range(self.ntries):			# For max number of tries
			nex[mode] = randint(self.sz[mode])	# Select a random sub/rel/obj

			if (nex[0], nex[2]) in self.counts and self.counts[(nex[0], nex[2])] > 0 and tuple(nex) not in self.xs:	
				res = (tuple(nex), -1.0)	# Assign negative label to the generated negative sample
				break
		return res


def type_index(xs):
	index = ddict(lambda: {0: set(), 1: set()})
	for i, j, k in xs:
		index[k][0].add(i)
		index[k][1].add(j)
	#for p, idx in index.items():
	#    print(p, len(idx[0]), len(idx[1]))
	return {k: {0: list(v[0]), 1: list(v[1])} for k, v in index.items()}
