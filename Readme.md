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

#### Usage:

##### Setup Environment:

* After installing python dependencies, execute `sh setup.sh` for setting up required things.

##### Start PPDB server:

* Running PPDB server is essential for running the main code.
* To start the server execute: `python ppdb/ppdb_server.py -port 9997`  (Let the server run in a separate terminal)

##### Run the main code:

* `python src/main_cesi.py -name reverb45_test_run`
* On executing the above command, all the output will be dumped in `output/reverb45_test_run` directory. 

##### Extra note:

* We recommend to view *.py files with tab size 8.
