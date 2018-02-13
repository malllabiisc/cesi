'''
Returns Glove embeddings for a phrase
'''

import numpy as np
import config, pdb
from pprint import pprint


# Get glove embeddings for phrases
def getGlove(phr_list, gtype):
	dims 		= int(gtype.split('_')[1])
	wrd_list 	= set()
	glove_vecs 	= dict()

	for phr in phr_list: wrd_list = wrd_list.union(phr.split())
	wrd_list = list(wrd_list)
	resp = config.glove[gtype].find({"_id": {"$in": wrd_list}})

	for ele in resp: glove_vecs[ele['_id']] = np.float32(ele['vec'])

	vec = np.zeros((len(phr_list), dims), np.float32)

	for i in range(len(phr_list)):
		phr = phr_list[i]

		if phr in glove_vecs:
			vec[i] = glove_vecs[phr]

		elif len(phr.split()) == 1:
			vec[i] = np.random.randn(dims)
		else:
			count = 0
			for wrd in phr.split():
				if wrd not in config.stpwords and wrd in glove_vecs:
					vec[i] += glove_vecs[wrd]
					count += 1

			if count == 0: 	vec[i] = np.random.randn(dims)
			else: 		vec[i] = vec[i] / count

	return vec
