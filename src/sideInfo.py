'''
Side Information Acquisition module
'''

from helper import *
from pprint import pprint
from nltk.corpus import wordnet
from nltk.wsd import lesk
import pdb, itertools
from unionFind import DisjointSet
from collections import defaultdict as ddict
import editdistance
from nltk.corpus import stopwords

'''*************************************** INPUT CLASS ********************************************'''
class SideInfo(object):
	def __init__(self, args, triples_list, amb_mentions, amb_ent, isAcronym):
		self.p 		= args
		self.file 	= open(self.p.out_path + '/side_info.txt', 'w')
		self.triples  	= triples_list
		self.stopwords 	= stopwords.words('english')

		self.initVariables()
		self.fixTypos(amb_ent, amb_mentions, isAcronym)
		self.process()

	def process(self):

		ent1List, relList, ent2List = [], [], []	# temp variables
		for triple in self.triples:			# Get all subject, objects and relations
			ent1List.append(triple['triple'][0])
			relList .append(triple['triple'][1])
			ent2List.append(triple['triple'][2])

		# Get unique list of subject, relations, and objects
		self.rel_list	= list(set(relList))
		self.ent_list	= list(set().union( list(set(ent1List)), list(set(ent2List))))
		self.sub_list   = list(set(ent1List))

		for ent in self.ent_list: self.clean_ent_list.append(ent.split('|')[0])

		# Generate a unique id for each entity and relations
		self.ent2id 	= dict([(v,k) for k,v in enumerate(self.ent_list)])
		self.rel2id 	= dict([(v,k) for k,v in enumerate(self.rel_list)])

		self.isSub = {}
		for sub in self.sub_list:
			self.isSub[self.ent2id[sub]] = 1

		# Get frequency of occurence of entities and relations
		for ele in ent1List:
			ent = self.ent2id[ele]
			self.ent_freq[ent] = self.ent_freq.get(ent, 0)
			self.ent_freq[ent] += 1

		for ele in ent2List:
			ent = self.ent2id[ele]
			self.ent_freq[ent] = self.ent_freq.get(ent, 0)
			self.ent_freq[ent] += 1

		for ele in relList:
			rel = self.rel2id[ele]			
			self.rel_freq[rel] = self.rel_freq.get(rel, 0)
			self.rel_freq[rel] += 1

		# Creating inverse mapping as well
		self.id2ent   	= invertDic(self.ent2id)
		self.id2rel	= invertDic(self.rel2id)

		# Store triples in the form of (sub_id, rel_id, obj_id)
		for triple in self.triples:
			trp = (self.ent2id[triple['triple'][0]], self.rel2id[triple['triple'][1]], self.ent2id[triple['triple'][2]])
			self.trpIds.append(trp)



	def wikiLinking(self):
		for trp in self.triples:
			sub, obj 	= trp['triple'][0], trp['triple'][2]
			sub_id, obj_id 	= self.ent2id[sub], self.ent2id[obj]


			if trp['ent_lnk_sub'] != None:
				self.ent2wiki[sub_id] = self.ent2wiki.get(sub_id, set())
				self.ent2wiki[sub_id].add(trp['ent_lnk_sub'])

			if trp['ent_lnk_obj'] != None:
				self.ent2wiki[obj_id] = self.ent2wiki.get(obj_id, set())
				self.ent2wiki[obj_id].add(trp['ent_lnk_obj'])

		for k, v in self.ent2wiki.items(): self.ent2wiki[k] = list(v)
	
		self.setHeading('Wikipedia Linking')
		self.printCluster(self.ent2wiki, self.id2ent, 'm2ol')
	
	def kbpLinking(self):
		for trp in self.triples:
			if trp['rel_info'] == []: continue

			rel 	= trp['triple'][1]
			rel_id 	= self.rel2id[rel]

			self.rel2kbp[rel_id] 	= self.rel2kbp.get(rel_id, set())
			for ele in trp['rel_info']:
				if ele[0] == None or ele[2] == None or ele[0] == 'O' or ele[2] == 'O': continue
				self.rel2kbp[rel_id].add('|'.join(ele))
		
		for k, v in self.rel2kbp.items(): self.rel2kbp[k] = list(v)

		self.setHeading('KBP Relation Clusters')
		self.printCluster(self.rel2kbp, self.id2rel, 'm2ol')

	def ppdbLinking(self):
		self.ent2ppdb = getPPDBclusters(self.p.ppdb_url, self.ent_list, self.ent2id)
		self.rel2ppdb = getPPDBclusters(self.p.ppdb_url, self.rel_list, self.rel2id)

		self.setHeading('PPDB Entity Clusters')	
		self.printCluster(self.ent2ppdb, self.id2ent, 'm2ol')

		self.setHeading('PPDB Relation Clusters')
		self.printCluster(self.rel2ppdb, self.id2rel, 'm2ol')

	def wordnetLinking(self):
		
		for trp in self.triples:
			sub, rel, obj 		  = trp['triple']
			raw_sub, raw_rel, raw_obj = trp['raw_triple']
			sub_id, rel_id, obj_id 	  = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]

			for sentence in trp['src_sentences']:
				# sent = [wrd.lower() for wrd in sentence.split()]
				sent = sentence.split()

				''' 92 is the length of list returned by dir when lesk is successful '''
				self.ent2wnet[sub_id] = self.ent2wnet.get(sub_id, set())
				res = lesk(sent, raw_sub) 
				if len(dir(res)) == 92: self.ent2wnet[sub_id].add(res.name())

				self.ent2wnet[obj_id] = self.ent2wnet.get(obj_id, set())
				res = lesk(sent, raw_obj) 
				if len(dir(res)) == 92: self.ent2wnet[obj_id].add(res.name())

				self.rel2wnet[rel_id] = self.rel2wnet.get(rel_id, set())
				res = lesk(sent, raw_rel) 
				if len(dir(res)) == 92: self.rel2wnet[rel_id].add(res.name())


		# for ent in self.ent_list: self.ent2wnet[self.ent2id[ent]] = [ele.name() for ele in lesk(ent)]
		# for rel in self.rel_list: self.rel2wnet[self.rel2id[rel]] = [ele.name() for ele in wordnet.synsets(rel)]

		self.setHeading('Wordnet Entity Clusters')
		self.printCluster(self.ent2wnet, self.id2ent, 'm2ol')

		self.setHeading('Wordnet Relation Clusters')
		self.printCluster(self.rel2wnet, self.id2rel, 'm2ol')


	def amieInfo(self):
		uf = DisjointSet()
		min_supp = 2
		min_conf = 0.2
		amie_cluster = []
		rel_so = {}

		for trp in self.triples:
			sub, rel, obj = trp['triple']
			sub_id, rel_id, obj_id = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]

			rel_so[rel_id] = rel_so.get(rel_id, set())
			rel_so[rel_id].add((sub_id, obj_id))

		for r1, r2 in itertools.combinations(rel_so.keys(), 2):
			supp = len(rel_so[r1].intersection(rel_so[r2]))
			if supp < min_supp: continue

			s1, _ = zip(*list(rel_so[r1]))
			s2, _ = zip(*list(rel_so[r2]))

			z_conf_12, z_conf_21 = 0, 0
			for ele in s1: 
				if ele in s2: z_conf_12 += 1 
			for ele in s2: 
				if ele in s1: z_conf_21 += 1

			conf_12 = supp / z_conf_12
			conf_21 = supp / z_conf_21

			if conf_12 >= min_conf and conf_21 >= min_conf:
				amie_cluster.append((r1, r2))		# Replace with union find DS
				uf.add(r1, r2)

		self.rel2amie = uf.leader

		self.setHeading('AMIE Relation Clusters')
		self.printCluster(uf.leader, self.id2rel, 'm2o')

	def morphNorm(self):

		for trp in self.triples:
			sub_n, rel_n, obj_n 	= trp['triple_norm']
			sub, rel, obj 		= trp['triple']
			subId, relId, objId 	= self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]

			if sub_n != 'NULL':
				self.ent2morph[subId] = self.ent2morph.get(subId, set())
				self.ent2morph[subId].add(sub_n)
			
			if obj_n != 'NULL':
				self.ent2morph[objId] = self.ent2morph.get(objId, set())
				self.ent2morph[objId].add(obj_n)
			
			self.rel2morph[relId] = self.rel2morph.get(relId, set())
			self.rel2morph[relId].add(rel_n)
		
		for k, v in self.ent2morph.items(): self.ent2morph[k] = list(v)
		for k, v in self.rel2morph.items(): self.rel2morph[k] = list(v)

		self.setHeading('MORPH NORM Entity Clusters')
		self.printCluster(self.ent2morph, self.id2ent, 'm2ol')

		self.setHeading('MORPH NORM Relation Clusters')
		self.printCluster(self.rel2morph, self.id2rel, 'm2ol')

	def tokenOverlap(self, amb_mentions, amb_ent):
		
		raw_ent_list = unique([ent.split('|')[0] for ent in self.ent_list])

		self.ent_term_freq = ddict(int)
		for ent in raw_ent_list:
			for tok in ent.split(): 
				if tok not in self.stopwords:
					self.ent_term_freq[tok] = self.ent_term_freq[tok]+1

		cluster = ddict(set)
		for ent in self.ent_list:
			val = ent.split('|')[0]
			for tok in val.split():
				if tok in self.stopwords: continue
				cluster[tok].add(ent)
		
		self.ent2idfTok = {}
		for rep, clust in cluster.items():
			for e1, e2 in itertools.combinations(clust, 2):
				
				ent1, ent2 = self.ent2id[e1], self.ent2id[e2]
				if (ent1, ent2) in self.ent2idfTok: continue

				e1, e2 = e1.split('|')[0], e2.split('|')[0]
				if e1 == e2: continue

				if e1 in amb_mentions or e2 in amb_mentions: continue
				# if (e1 in amb_mentions and e2 in amb_mentions) and (e1 not in amb_ent and e2 not in amb_ent): continue

				tokens1 = set(e1.split()) - set(self.stopwords);
				tokens2 = set(e2.split()) - set(self.stopwords);

				intersect = tokens1.intersection(tokens2)
				union 	  = tokens1.union(tokens2)

				num, den = 0.0, 0.0;
				for ele in intersect: 	num += 1.0 / np.log(1 + self.ent_term_freq[ele]);
				for ele in union:	den += 1.0 / np.log(1 + self.ent_term_freq[ele]);

				score = num / den if den != 0 else 0
				if score != 0:
					self.ent2idfTok[(ent1, ent2)] = score
				
		

		self.rel_term_freq = ddict(int)
		for rel in self.rel_list:
			for tok in rel.split(): 
				self.rel_term_freq[tok] = self.rel_term_freq[tok]+1 if tok not in self.stopwords else self.rel_term_freq[tok]

		self.rel2idfTok = {}

		cluster = ddict(set)
		for rel in self.rel_list:
			for tok in rel.split():
				if tok in self.stopwords: continue
				cluster[tok].add(rel)

		for rep, clust in cluster.items():
			for r1, r2 in itertools.combinations(clust, 2):
				id1, id2 = self.rel2id[r1], self.rel2id[r2]
				if (id1, id2) in self.rel2idfTok: continue

				tokens1 = set(r1.split()) - set(self.stopwords);
				tokens2 = set(r2.split()) - set(self.stopwords);

				intersect = tokens1.intersection(tokens2)
				union 	  = tokens1.union(tokens2)

				num, den = 0.0, 0.0;
				for ele in intersect: 	num += 1.0 / np.log(1 + self.rel_term_freq[ele]);
				for ele in union:	den += 1.0 / np.log(1 + self.rel_term_freq[ele]);

				score = num / den if den != 0 else 0
				if score != 0:
					self.rel2idfTok[(id1, id2)] = score
	
		self.setHeading('IDF TOKEN Entity Clusters')
		for (e1, e2), scr in self.ent2idfTok.items():
			if e1 in self.isSub and e2 in self.isSub:
				self.file.write('(%s, %s) -> %f\n' % (self.id2ent[e1], self.id2ent[e2], scr))

		self.setHeading('IDF TOKEN Relation Clusters')
		for (r1, r2), scr in self.rel2idfTok.items():
			self.file.write( '(%s, %s) -> %f\n' % (self.id2rel[r1], self.id2rel[r2], scr))

	def lnkEntTypeInfo(self):
		self.coarseLnkEntClust, self.fineLnkEntClust = getLnkEntTypeClusters(self.wiki2ent)

	''' PRINT FUNCTIONS '''
	def setHeading(self, title):
		self.file.write( '\n' + '*'*30 + title + '*'*30 + '\n')

	def printCluster(self, id2rep, id2name, ctype = 'm2o'):
		inv_dict = invertDic(id2rep, ctype)

		for rep, cluster in inv_dict.items():
			cluster = [id2name[ele].split('|')[0] for ele in cluster if ele in self.isSub or id2name == self.id2rel]
			cluster = list(set(cluster))

			if len(cluster) <= 1: continue

			self.file.write(str(rep) + '\n')
			for ele in cluster: 
				self.file.write('\t' + ele + '\n')

	def fixTypos(self, amb_ent, amb_mentions, isAcronym):
		ent2wiki = ddict(set)
		uf 	 = DisjointSet()

		''' Use Wiki Info to correct typos '''
		for trp in self.triples:
			sub, obj 	= trp['triple'][0], trp['triple'][2]
			sub = sub.split('|')[0]

			if trp['ent_lnk_sub'] != None: ent2wiki[sub].add(trp['ent_lnk_sub'])
			if trp['ent_lnk_obj'] != None: ent2wiki[obj].add(trp['ent_lnk_obj'])

		wiki2ent = invertDic(ent2wiki, 'm2os')

		for wiki, clust in wiki2ent.items():
			for e1, e2 in itertools.combinations(clust, 2):
				if e1 == e2: continue
				if e1 == '' or e2 == '': continue
				if e1 in amb_ent or e2 in amb_ent: continue

				if editdistance.eval(e1, e2) <= 2:
					uf.add(e1, e2)
				elif e1 in isAcronym or e2 in isAcronym:
					uf.add(e1, e2)

		''' Use Wnet Info to correct typos '''
		wnet2ent = ddict(set)
		for trp in self.triples:
			sub, rel, obj 		  = trp['triple']
			sub = sub.split('|')[0]
			raw_sub, raw_rel, raw_obj = trp['raw_triple']

			for sentence in trp['src_sentences']:
				sent = sentence.split()

				res = wordnetDisamb(sent, raw_sub)
				if res != None: wnet2ent[res].add(sub)

				res = wordnetDisamb(sent, raw_obj)
				if res != None: wnet2ent[res].add(obj)
		
		for wnet, clust in wnet2ent.items():
			for e1, e2 in itertools.combinations(clust, 2):
				if e1 == e2: continue
				if e1 == '' or e2 == '': continue
				if e1 in amb_ent or e2 in amb_ent: continue
				if len(e1) <= 3 or len(e2) <= 3: continue

				if editdistance.eval(e1, e2) <= 2:
					uf.add(e1, e2)
				elif e1 in isAcronym or e2 in isAcronym:
					uf.add(e1, e2)
		
		''' Use Morph Info to correct typos '''
		morph2ent = ddict(set)

		for trp in self.triples:
			sub_n, rel_n, obj_n 	= trp['triple_norm']
			sub, rel, obj 		= trp['triple']
			sub = sub.split('|')[0]

			if sub_n != 'NULL': morph2ent[sub_n].add(sub)
			if obj_n != 'NULL': morph2ent[obj_n].add(obj)
		
		for morph, clust in morph2ent.items():
			for e1, e2 in itertools.combinations(clust, 2):
				if e1 == e2: continue
				if e1 == '' or e2 == '': continue
				if e1 in amb_ent or e2 in amb_ent: continue

				if editdistance.eval(e1, e2) <= 2:
					uf.add(e1, e2)
				elif e1 in isAcronym or e2 in isAcronym:
					uf.add(e1, e2)


		''' Using PPDB Information '''
		sub_list = set()
		for trp in self.triples: 
			sub_list.add(trp['triple'][0].split('|')[0])
			sub_list.add(trp['triple'][2].split('|')[0])

		sub_list = list(sub_list)
		ent2ppdb = getPPDBclustersRaw(self.p.ppdb_url, sub_list)
		ppdb2ent = invertDic(ent2ppdb, 'm2ol')

		for _, clust in ppdb2ent.items():
			for e1, e2 in itertools.combinations(clust, 2):
				if e1 == e2: continue
				if e1 == '' or e2 == '': continue
				if len(e1) <= 3 or len(e2) <= 3: continue
				if e1 in amb_ent or e2 in amb_ent: continue

				if editdistance.eval(e1, e2) <= 2:
					uf.add(e1, e2)
				elif e1 in isAcronym or e2 in isAcronym:
					uf.add(e1, e2)


		ent2rep = {}
		for rep, clust in uf.group.items():
			rep = max(clust, key=len)
			for ele in clust: ent2rep[ele] = rep

		''' FIX TYPOS '''
		for i, trp in enumerate(self.triples):
			sub, rel, obj = trp['triple']

			if sub in ent2rep: 	sub = ent2rep[sub]
			else: 			sub = ' '.join([ent2rep.get(ele, ele) for ele in sub.split()])

			if obj in ent2rep: 	obj = ent2rep[obj]
			else: 			obj = ' '.join([ent2rep.get(ele, ele) for ele in obj.split()])

			self.triples[i]['triple'] = [sub, rel, obj]

		rep2ent = invertDic(ent2rep, 'm2o')
		self.setHeading('TYPO DETECTION')
		for rep, clust in rep2ent.items():
			self.file.write(rep + '\n')
			for ele in clust:
				self.file.write('\t' + ele + '\n')


	''' ATTRIBUTES DECLARATION '''
	def initVariables(self):
		self.ent_list = None		# List of all entities
		self.clean_ent_list = []
		self.rel_list = None		# List of all relations
		self.trpIds = []		# List of all triples in id format

		self.ent2id = None		# Maps entity to its id (o2o)
		self.rel2id = None		# Maps relation to its id (o2o)
		self.id2ent = None		# Maps id to entity (o2o)
		self.id2rel = None		# Maps id to relation (o2o)

		self.rel_pairs = None		# List of all relation pairs

		self.uf = DisjointSet()

		self.ent_freq = {}		# Entity to its frequency
		self.rel_freq = {}		# Relation to its frequency

		self.ent2clust	= {}
		self.rel2clust	= {}

		self.final_facts = []

		self.ent2wiki = {}		# Entity to Wiki Id
		self.ent2ppdb = {}		# PPDB entity clusters
		self.rel2ppdb = {}		# PPDB relation clusters
		self.wiki2ent = None		# Wiki Id to entity

		self.ent2morph = {}
		self.rel2morph = {}

		self.ent2wnet = {}
		self.rel2wnet = {}

		self.rel2kbp = {}
		self.rel2amie = {}

		''' Type Info '''
		self.trpTypeInfo		= []
		self.coarseLnkEntClust 	= None	# Linked Entity Coarse type clusters
		self.fineLnkEntClust 	= None	# Linked Entity Fine type clusters
		# self.lnkEntTypeInfo()