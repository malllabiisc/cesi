import sys
import numpy as np
from numpy import sqrt, squeeze, zeros_like
from numpy.random import randn, uniform


def init_unif(sz):
	"""
	Uniform intialization -> [-1/sqrt(N), 1/sqrt(N)], sz = (N, N, M)

	Heuristic commonly used to initialize deep neural networks
	"""
	bnd = 1 / sqrt(sz[0])
	p = uniform(low=-bnd, high=bnd, size=sz)
	return squeeze(p)				# Remove single-dimensional entries from the shape of an array.


def init_nunif(sz):
	"""
	Normalized uniform initialization -> [ -sqrt(6)/sqrt(N+N), sqrt(6)/sqrt(N+N) ]

	See Glorot X., Bengio Y.: "Understanding the difficulty of training
	deep feedforward neural networks". AISTATS, 2010
	"""
	bnd = sqrt(6) / sqrt(sz[0] + sz[1])
	p = uniform(low=-bnd, high=bnd, size=sz)
	return squeeze(p)

# Normal Initialization
def init_randn(sz):
	return squeeze(randn(*sz))

import pdb

# This class is an abstraction over np.ndarray
class Parameter(np.ndarray): 

	# This method is called before the creation of any instance (before __init__ method)
	def __new__(cls, *args, **kwargs):
		# TODO: hackish, find better way to handle higher-order parameters
		value 	= kwargs.pop('value', None)

		if type(value) != np.ndarray:
			if len(args[0]) == 3:
				sz = (args[0][1], args[0][2]) 	# N x M 
				arr = np.array( [ Parameter._init_array(sz, args[1]) for _ in range(args[0][0]) ]) # Loop executed: N times, In each iteration based on the type of 
			else:
				arr = Parameter._init_array(args[0], args[1])	# Init with 2d array
		else:
			arr = value

		arr = arr.view(cls) 			# changes type of arr from 'np.ndarray' to 'skge.param.Parameter'
		arr.name = kwargs.pop('name', None)	# Assign a name if passed as argument
		arr.post = kwargs.pop('post', None)	# Assign a post processing step after each update

		if arr.post is not None:		# Appy post processing step is given
			arr = arr.post(arr)

		return arr

	def __array_finalize__(self, obj):		# Copies name and post processing method from obj to self
		if obj is None:
			return
		self.name = getattr(obj, 'name', None)	# Equivalent to obj.name
		self.post = getattr(obj, 'post', None)	# Equivalent to obj.post

	@staticmethod
	def _init_array(shape, method):
		mod = sys.modules[__name__]		# Prove access to methods declared
		method = 'init_%s' % method 		# Construct the name of the the intialization method
		if not hasattr(mod, method): 	raise ValueError('Unknown initialization (%s)' % method)
		elif len(shape) != 2: 		raise ValueError('Shape must be of size 2')
		return getattr(mod, method)(shape)	# Call the asked method for initialization

# Base class for all optimizers
class ParameterUpdate(object):

	def __init__(self, param, learning_rate):
		self.param = param
		self.learning_rate = learning_rate

	def __call__(self, gradient, idx=None):		# Allows instance to be called as a function
		self._update(gradient, idx)		# Update params based on gradient values
		if self.param.post is not None:		# Apply post processing step after each update
			self.param = self.param.post(self.param, idx)

	def reset(self):
		pass


class SGD(ParameterUpdate):
	"""
	Class to perform SGD updates on a parameter
	"""

	def _update(self, g, idx):
		self.param[idx] -= self.learning_rate * g


class AdaGrad(ParameterUpdate):
	"""
	Class to perform AdaGrad updates on a parameter
	"""

	def __init__(self, param, learning_rate):
		super(AdaGrad, self).__init__(param, learning_rate)
		self.p2 = zeros_like(param)		# Returns an array of zeros with the same shape and type

	def _update(self, g, idx=None):
		self.p2[idx] += g * g
		H = np.maximum(np.sqrt(self.p2[idx]), 1e-7)
		self.param[idx] -= self.learning_rate * g / H

	def reset(self):
		self.p2 = zeros_like(self.p2)


def normalize(M, idx=None):
	if idx is None:
		M = M / np.sqrt(np.sum(M ** 2, axis=1))[:, np.newaxis]
	else:
		nrm = np.sqrt(np.sum(M[idx, :] ** 2, axis=1))[:, np.newaxis]
		M[idx, :] = M[idx, :] / nrm
	return M


# Normalize entries except those rows whose sum of square is less than 1
def normless1(M, idx=None):
	nrm = np.sum(M[idx] ** 2, axis=1)[:, np.newaxis]
	nrm[nrm < 1] = 1		# Entries less than 1 are not normalized
	M[idx] = M[idx] / nrm
	return M
