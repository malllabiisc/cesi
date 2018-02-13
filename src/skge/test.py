import sys
import numpy as np
from numpy import sqrt, squeeze, zeros_like
from numpy.random import randn, uniform

import pdb

def init_unif(sz):
	bnd = 1 / sqrt(sz[0])
	p = uniform(low=-bnd, high=bnd, size=sz)
	return squeeze(p)

sz = [10,10,5]
a = init_unif(sz)
pdb.set_trace()