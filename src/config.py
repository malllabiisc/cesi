'''
File containing all global variables used in the Entice code.
NOTE: This file needs to be removed for scaling Entice to multiple users.
'''

from enum import Enum
from helper import *
from pymongo import MongoClient
from nltk.corpus import stopwords

# Server configurations
time_info 	= None
dpath 		= ''
max_threads	= 15
embed_dims	= None

# Service URLs
glove_url 	= 'http://10.24.28.104:9999/'
ppdb_url	= 'http://10.24.28.104:9997/'


# File Locations
file_triples		= '/triples.txt'
file_timeinfo		= '/time_info.txt'
file_entEmbed 		= '/embed_ent.pkl'
file_relEmbed		= '/embed_rel.pkl'
file_entClust		= '/cluster_ent.txt'
file_relClust		= '/cluster_rel.txt'
file_sideinfo		= '/side_info.txt'
file_sideinfo_pkl	= '/side_info.pkl'
file_hyperparams	= '/hyperparams.txt'
file_embedInit		= '/embed_init.txt'
file_evalResults	= '/final_results.txt'

stpwords = stopwords.words('english')