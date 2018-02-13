'''
Implementation of different metrics used for evaluating CESI results

C: Clusters produced by algorithm
E: Gold standard cluster
'''

import itertools, sys

def macroPrecision(C_clust2ele, E_ele2clust):
	num_prec = 0

	for _, cluster in C_clust2ele.items():
		isFirst = True
		res = set()
		for ele in cluster:
			if ele not in E_ele2clust: 
				# sys.stdout.write('.')
				continue
			if isFirst:
				res = E_ele2clust[ele]
				isFirst = False
				continue

			res = res.intersection(E_ele2clust[ele])

		if   len(res) == 1: num_prec += 1
		elif len(res) > 1: print 'ERROR In Clustering micro!!!'

	if len(C_clust2ele) == 0: return 0
	return float(num_prec) / float(len(C_clust2ele))

def microPrecision(C_clust2ele, E_ele2clust):
	num_prec = 0
	total = 0

	for _, cluster in C_clust2ele.items():
		freq_map = {}
		total += len(cluster)

		for ent in cluster:
			if ent not in E_ele2clust: 
				# sys.stdout.write('.')
				continue
			for ele in E_ele2clust[ent]:
				freq_map[ele] = freq_map.get(ele, 0)
				freq_map[ele] += 1
		max_rep = 0
		for k, v in freq_map.items(): max_rep = max(max_rep, v)

		num_prec += max_rep

	if total == 0: return 0
	return float(num_prec) / float(total)

def pairPrecision(C_clust2ele, E_ele2clust):
	num_hit = 0
	num_pairs = 0

	for _, cluster in C_clust2ele.items():
		all_pairs = list(itertools.combinations(cluster, 2))
		num_pairs += len(all_pairs)

		for e1, e2 in all_pairs:
			if e1 not in E_ele2clust or e2 not in E_ele2clust:
				# sys.stdout.write('.')
				continue

			res = E_ele2clust[e1].intersection(E_ele2clust[e2])
			if   len(res) == 1: num_hit += 1
			# elif len(res) > 1: print 'ERROR In Clustering pairwise!!!'

	if num_pairs == 0: return 0
	return float(num_hit) / float(num_pairs)

def pairwiseMetric(C_clust2ele, E_ele2clust, E_clust2ent):
	num_hit = 0
	num_C_pairs = 0
	num_E_pairs = 0

	for _, cluster in C_clust2ele.items():
		all_pairs = list(itertools.combinations(cluster, 2))
		num_C_pairs += len(all_pairs)

		for e1, e2 in all_pairs:
			if e1 in E_ele2clust and e2 in E_ele2clust and len(E_ele2clust[e1].intersection(E_ele2clust[e2])) > 0: num_hit += 1

	for rep, cluster in E_clust2ent.items(): 
		num_E_pairs += len(list(itertools.combinations(cluster, 2)))

	if num_C_pairs == 0 or num_E_pairs == 0: 
		return 1e-6, 1e-6

	# print num_hit, num_C_pairs, num_E_pairs
	return float(num_hit) / float(num_C_pairs), float(num_hit) / float(num_E_pairs)


def calcF1(prec, recall):
	if prec + recall == 0: return 0
	return 2 * (prec * recall) / (prec + recall)

def microF1(C_ele2clust, C_clust2ele, E_ele2clust, E_clust2ent):
	micro_prec 	= microPrecision(C_clust2ele, E_ele2clust)
	micro_recall 	= microPrecision(E_clust2ent, C_ele2clust)
	micro_f1	= calcF1(micro_prec, micro_recall)
	return micro_f1

def macroF1(C_ele2clust, C_clust2ele, E_ele2clust, E_clust2ent):
	macro_prec 	= macroPrecision(C_clust2ele, E_ele2clust)
	macro_recall 	= macroPrecision(E_clust2ent, C_ele2clust)
	macro_f1	= calcF1(macro_prec, macro_recall)
	return macro_f1


def pairF1(C_ele2clust, C_clust2ele, E_ele2clust, E_clust2ent):
	pair_prec,pair_recall = pairwiseMetric(C_clust2ele, E_ele2clust, E_clust2ent)
	pair_f1		= calcF1(pair_prec, pair_recall)
	return pair_f1

def evaluate(C_ele2clust, C_clust2ele, E_ele2clust, E_clust2ent):
	macro_prec 	= macroPrecision(C_clust2ele, E_ele2clust)
	macro_recall 	= macroPrecision(E_clust2ent, C_ele2clust)
	macro_f1	= calcF1(macro_prec, macro_recall)

	micro_prec 	= microPrecision(C_clust2ele, E_ele2clust)
	micro_recall 	= microPrecision(E_clust2ent, C_ele2clust)
	micro_f1	= calcF1(micro_prec, micro_recall)

	pair_prec,pair_recall = pairwiseMetric(C_clust2ele, E_ele2clust, E_clust2ent)
	pair_f1		= calcF1(pair_prec, pair_recall)

	pairx_prec 	= pairPrecision(C_clust2ele, E_ele2clust)
	pairx_recall 	= pairPrecision(E_clust2ent, C_ele2clust)
	pairx_f1	= calcF1(pairx_prec, pairx_recall)

	return {
		'macro_prec': 	round(macro_prec, 	4),
		'macro_recall':	round(macro_recall, 	4),
		'macro_f1':	round(macro_f1, 	4),

		'micro_prec': 	round(micro_prec, 	4),
		'micro_recall':	round(micro_recall, 	4),
		'micro_f1':	round(micro_f1, 	4),

		'pair_prec': 	round(pair_prec, 	4),
		'pair_recall':	round(pair_recall, 	4),
		'pair_f1':	round(pair_f1, 		4),

		'pairx_prec': 	round(pairx_prec, 	4),
		'pairx_recall':	round(pairx_recall, 	4),
		'pairx_f1':	round(pairx_f1, 	4)
	}