import numpy as np
from numpy.random import shuffle
from collections import defaultdict
from skge.param import Parameter, AdaGrad
import timeit
import pickle, pdb
# from memory_profiler import profile

_cutoff = 30

_DEF_NBATCHES = 100
_DEF_POST_EPOCH = []
_DEF_LEARNING_RATE = 0.1
_DEF_SAMPLE_FUN = None
_DEF_MAX_EPOCHS = 1000
_DEF_MARGIN = 1.0


class Config(object):

	def __init__(self, model, trainer):
		self.model = model
		self.trainer = trainer

	def __getstate__(self):
		return {
			'model': self.model,
			'trainer': self.trainer
		}

class Model(object):
	"""
	Base class for all Knowledge Graph models
	Implements basic setup routines for parameters and serialization methods
	Subclasses need to implement:
	- scores(self, ss, ps, os)
	- _gradients(self, xys) for StochasticTrainer
	- pairwise_gradients(self, pxs, nxs) for PairwiseStochasticTrainer
	"""

	def __init__(self, *args, **kwargs):
		#super(Model, self).__init__(*args, **)
		self.params = {}					 # Paramters to be learned
		self.hyperparams = {}					 # Dictionary of hyper parameters

	def add_param(self, param_id, shape, post=None, value=None):
		value = Parameter(shape, self.init, name=param_id, post=post, value=value)
		setattr(self, param_id, value)
		self.params[param_id] = value				# Make entry of the parameter in params list

	def add_hyperparam(self, param_id, value):			# Adds a hyperparameter
		setattr(self, param_id, value)				# self.paramid = value
		self.hyperparams[param_id] = value

	def __getstate__(self):						# Returns parameter and hyperparams
		return {
			'hyperparams': self.hyperparams,
			'params': self.params
		}

	def __setstate__(self, st):					# Set params and hyperparams from prev saved state
		self.params = {}
		self.hyperparams = {}
		for pid, p in st['params'].items():
			self.add_param(pid, None, None, value=p)
		for pid, p in st['hyperparams'].items():
			self.add_hyperparam(pid, p)

	def save(self, fname, protocol=pickle.HIGHEST_PROTOCOL):	# Dump the object to a file
		with open(fname, 'wb') as fout:
			pickle.dump(self, fout, protocol=protocol)

	@staticmethod
	def load(fname):						# Retrieve the object from the file
		with open(fname, 'rb') as fin:
			mdl = pickle.load(fin)
		return mdl


class StochasticTrainer(object):
	"""
	Stochastic gradient descent trainer with scalar loss function.
	Models need to implement
	gradients(self, xys)
	to be trained with this class.
	"""

	def __init__(self, *args, **kwargs):
		self.model 		= args[0]								# Takes model as the first argument
		self.hyperparams 	= {}									# List of all hyper paramaters
		self.add_hyperparam('max_epochs', 	kwargs.pop('max_epochs', _DEF_MAX_EPOCHS))		# Set the number of epochs
		self.add_hyperparam('nbatches', 	kwargs.pop('nbatches', _DEF_NBATCHES))			# Number of batches
		self.add_hyperparam('learning_rate', 	kwargs.pop('learning_rate', _DEF_LEARNING_RATE))	# Learning rage

		self.post_epoch 	= kwargs.pop('post_epoch', _DEF_POST_EPOCH)				# Set callback; executed after ever epoch
		self.samplef 		= kwargs.pop('samplef', _DEF_SAMPLE_FUN)				# sample() function of sampler: Random/LCAW/Corruption (default=None)
		pu 			= kwargs.pop('param_update', AdaGrad)					# Choose optimizer (default = AdaGrad)
		self._updaters 		= {key: pu(param, self.learning_rate) for key, param in self.model.params.items()} # Creating instances of ParameterUpdate Object

	def __getstate__(self):					# Get the list of hyperparams
		return self.hyperparams

	def __setstate__(self, st):				# Set the hyperparams
		for pid, p in st['hyperparams']:
			self.add_hyperparam(pid, p)

	def add_hyperparam(self, param_id, value):		# Add a hyperparam
		setattr(self, param_id, value)
		self.hyperparams[param_id] = value

	def fit(self, xs, ys):					# Start Learning parameters
		self.optim(list(zip(xs, ys)))

	def pre_epoch(self):					# Loss accumulator
		self.loss = 0

	# @profile
	def optim(self, xys):					# Performs actual optimization
		idx 		= np.arange(len(xys))					# Index for every triple in dataset
		self.batch_size = int(np.ceil(len(xys) / self.nbatches))	  	# Calculte batch size (n_obsv / n_batches)
		batch_idx 	= np.arange(self.batch_size, len(xys), self.batch_size) # np.arange(start, stop, step) -> To get split positions (10,50,10) = [10,20,30,40]

		for self.epoch in range(1, self.max_epochs + 1): 	# Running for maximum number of epochs
			# shuffle training examples
			self.pre_epoch()				# Set loss = 0
			shuffle(idx)					# Shuffle the indexes of triples

			# store epoch for callback
			self.epoch_start = timeit.default_timer()	# Measuring time

			# process mini-batches
			for batch in np.split(idx, batch_idx):		# Get small subset of triples from training data
				bxys = [xys[z] for z in batch]		# Get triples present in the selected batch
				self.process_batch(bxys)		# Perform SGD using them

			# check callback function, if false return
			for f in self.post_epoch:			# Perform post epoch operation is specified
				if not f(self): break

	def process_batch(self, xys):
		# if enabled, sample additional examples
		if self.samplef is not None:				# Generate negative samples
			xys += self.samplef(xys)			# sample() of sampler is called (will generate n neg. samples for each positive sample)

		if hasattr(self.model, 'prepare_batch_step'):		# Do some pre-processsing if specified
			self.model.prepare_batch_step(xys)

		# take step for batch
		grads = self.model.gradients(xys)			# Compute Gradient requred for making the update
		self.loss += self.model.loss 				# Accumulate loss
		self.batch_step(grads)					# Update params

	def batch_step(self, grads):					# Updates params
		for paramID in self._updaters.keys():
			self._updaters[paramID](*grads[paramID])


class PairwiseStochasticTrainer(StochasticTrainer):	# Inheriting StochasticTrainer class
	"""
	Stochastic gradient descent trainer with pairwise ranking loss functions.
	Models need to implement
	pairwise_gradients(self, pxs, nxs)
	to be trained with this class.
	"""

	def __init__(self, *args, **kwargs):
		super(PairwiseStochasticTrainer, self).__init__(*args, **kwargs)
		self.model.add_hyperparam('margin', kwargs.pop('margin', _DEF_MARGIN))		# Hyperparam -> margin (gamma in paper)

	def fit(self, xs, ys):
		if self.samplef is None:
			pidx = np.where(np.array(ys) == 1)[0]		# Returns the indexes of positive triples
			nidx = np.where(np.array(ys) != 1)[0]		# Returns the indexes of negative triples
			pxs = [xs[i] for i in pidx]			# Collecting all triples from positive triples
			self.nxs = [xs[i] for i in nidx]		# self.nxs: All triples with negative label
			self.pxs = int(len(self.nxs) / len(pxs)) * pxs	# Doing some sort of scaling
			xys = list(range(min(len(pxs), len(self.nxs))))
			self.optim(xys)
		else:
			self.optim(list(zip(xs, ys)))			# Follow like Stochastic optmizer

	def pre_epoch(self):
		self.nviolations = 0
		if self.samplef is None:
			shuffle(self.pxs)
			shuffle(self.nxs)

	def process_batch(self, xys):
		pxs = []
		nxs = []

		for xy in xys:						# For all triples in batch
			if self.samplef is not None:			
				for nx in self.samplef([xy]):		# If sampler is defined then generate negative samples
					pxs.append(xy)			# We are repeatedly adding positive with its corresponding negative examples 
					nxs.append(nx)
			else:
				pxs.append((self.pxs[xy], 1))
				nxs.append((self.nxs[xy], 1))

		# take step for batch
		if hasattr(self.model, 'prepare_batch_step'):		# Do some pre-processsing if specified
			self.model.prepare_batch_step(pxs, nxs)

		grads = self.model.pairwise_gradients(pxs, nxs)	# Comupte gradients required for update

		# update if examples violate margin
		if grads is not None:
			self.nviolations += self.model.nviolations	# Just collects the count of number of violations
			self.batch_step(grads)				# Make the required updates

# Not used anywhere 
def sigmoid(fs):
	# compute elementwise gradient for sigmoid
	for i in range(len(fs)):
		if fs[i] > _cutoff:
			fs[i] = 1.0
		elif fs[i] < -_cutoff:
			fs[i] = 0.0
		else:
			fs[i] = 1.0 / (1 + np.exp(-fs[i]))
	return fs[:, np.newaxis]