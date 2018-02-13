## CESI: Canonicalizing Open Knowledge Bases using Embeddings and Side Information

This is the code for the paper [CESI: Canonicalizing Open Knowledge Bases using Embeddings and Side Information]().

#### Dependencies

* Python 2.7
* Rest can be installed using `requirements.txt`


#### Datasets

Datasets can be downloaded from: [Base](https://doc-0c-ao-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/pctniiusaca820lac9grgm0amt0rpgqf/1511344800000/14110994264170927396/*/1seaeutMYiRa1vI6wWQTlkkbyiu2VZu0b?e=download), [Ambiguous](https://doc-0s-ao-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/gd4jh5dq4s03m4ubuh5usc59ujj1rrin/1511344800000/14110994264170927396/*/1yNuhoRvxe6SOPebBgjG8f2fSXdi4-H4W?e=download), [Reverb45k](https://doc-14-ao-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/s0q1qpbpebj384fna543n6s94asher2k/1511344800000/14110994264170927396/*/1belXFXuIUApht18RX-abdkxsGeWOrKUe?e=download). 

After downloading, extract the datasets in `data` folder.

Preprocessed PPDB dataset can be downloaded from [link]().

#### Running the code

##### Start PPDB server:

`python ppdb_server.py <port[6686]>` 

##### Run the main code:

`python main_cesi.py <dataset[base/ambiguous/reverb45k]> <split[valid/test]>`

