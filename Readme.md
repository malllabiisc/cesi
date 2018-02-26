## CESI: Canonicalizing Open Knowledge Bases using Embeddings and Side Information

Source code and dataset for The WebConf 2018 (WWW 2018) paper: [CESI: Canonicalizing Open Knowledge Bases using Embeddings and Side Information]().

#### Dependencies

* Compatible with both Python 2.7/3.x
* Dependencies can be installed using `requirements.txt`


#### Datasets

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
	"entity_linking": {
		"subject":  "Frederick,_Maryland",
		"object":   "Alessandria"
	},
	"true_link": {
		"subject": "/m/09w_9",
		"object":  "/m/02bb_4"
	},
	"kbp_info": [],
	"src_sentences": [
		"Frederick had reached Alessandria",
		"By late October, Frederick had reached Alessandria."
	],
}        
```

Preprocessed PPDB dataset can be downloaded from [link]().

#### Usage:

##### Start PPDB server:

`python ppdb_server.py <port[6686]>` 

##### Run the main code:

`python main_cesi.py <dataset[base/ambiguous/reverb45k]> <split[valid/test]>`

