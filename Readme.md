## CESI: Canonicalizing Open Knowledge Bases using Embeddings and Side Information

Source code and dataset for [The WebConf 2018 (WWW 2018)](https://www2018.thewebconf.org/) paper: [CESI: Canonicalizing Open Knowledge Bases using Embeddings and Side Information](http://malllabiisc.github.io/publications/papers/cesi_www18.pdf).

![](https://raw.githubusercontent.com/malllabiisc/cesi/master/overview.png)
*CESI first acquires side information of noun and relation phrases of Open KB triples. In the second step, it learns embeddings of these NPs and relation phrases while utilizing the side information obtained in previous step. In the third step, CESI performs clustering over the learned embeddings to canonicalize NP and relation phrases. Please refer paper for more details*

### Dependencies

* Compatible with both Python 2.7/3.x
* Dependencies can be installed using `requirements.txt`


### Datasets

* Dataset ReVerb45k is included with the repository.
* The input to CESI is a KG as list of triples. Each triple is stored as a json in a new line. An example entry is shown below:

```json
{
	"_id": 	  36952,
	"triple": [
		"Frederick",
		"had reached",
		"Alessandria"
	],
	"triple_norm": [
		"frederick",
		"have reach",
		"alessandria"
	],
  	"true_link": {
		"subject": "/m/09w_9",
		"object":  "/m/02bb_4"
	},
  	"src_sentences": [
		"Frederick had reached Alessandria",
		"By late October, Frederick had reached Alessandria."
	],
	"entity_linking": {
		"subject":  "Frederick,_Maryland",
		"object":   "Alessandria"
	},
	"kbp_info": []
}        
```

* `_id` unique id of each triple in the Knowledge Graph. 
* `triple` denotes the actual triple in the Knowledge Graph
* `triple_norm` denotes the normalized form of the triple (after lemmatization, lower casing ...)
* `true_link` is the gold canonicalization of subject and object. For relations gold linking is not available.
* `src_sentences` is the list of sentences from which the triple was extracted by Open IE algorithms. 
* `entity_linking` is the Entity Linking side information which is utilized by CESI.
* `kbp_info` Knowledge-Base Propagation side information used by CESI.

### Usage:

##### Setup Environment:

* After installing python dependencies, execute `sh setup.sh` for setting up required things.
* Pattern library is required to run the code. Please install it from [Python 2.x](https://github.com/clips/pattern)/[Python 3.x](https://github.com/pattern3/pattern).

##### Start PPDB server:

* Running PPDB server is essential for running the main code.
* To start the server execute: `python ppdb/ppdb_server.py -port 9997`  (Let the server run in a separate terminal)

##### Run the main code:

* `python src/cesi_main.py -name reverb45_test_run`
* On executing the above command, all the output will be dumped in `output/reverb45_test_run` directory. 
* `-name` is an arbitrary name assigned to the run.

##### Extra note:

* We recommend to view *.py files with tab size 8.

### Citing:

```tex
@inproceedings{Vashishth:2018:CCO:3178876.3186030,
	author = {Vashishth, Shikhar and Jain, Prince and Talukdar, Partha},
	title = {CESI: Canonicalizing Open Knowledge Bases Using Embeddings and Side Information},
	booktitle = {Proceedings of the 2018 World Wide Web Conference},
	series = {WWW '18},
	year = {2018},
	isbn = {978-1-4503-5639-8},
	location = {Lyon, France},
	pages = {1317--1327},
	numpages = {11},
	url = {https://doi.org/10.1145/3178876.3186030},
	doi = {10.1145/3178876.3186030},
	acmid = {3186030},
	publisher = {International World Wide Web Conferences Steering Committee},
	address = {Republic and Canton of Geneva, Switzerland},
	keywords = {canonicalization, knowledge graph embeddings, knowledge graphs, open knowledge bases},
}
```
