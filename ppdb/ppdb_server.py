from flask import Flask, request
import json, argparse
app = Flask(__name__)

parser = argparse.ArgumentParser(description='')
parser.add_argument('-port', 	dest='port', 	type=int,	default='9997')
parser.add_argument('-src', 	dest='src_file', 		default='./ppdb/xxxl_clusters.txt')
args = parser.parse_args()

clusters = dict()

with open(args.src_file, 'r') as f:
	for line in f:
		if '\t' not in line:
			rep = line.strip()
		else:
			phr = line.strip()
			if phr not in clusters: clusters[phr] = [rep]
			else:
				if rep not in clusters[phr]: clusters[phr].append(rep)

def getRep(phr):
	phr = phr.lower()
	if phr in clusters: return clusters[phr]
	else: None

@app.route('/ppdb', methods=['GET', 'POST'])
def ppdb():
	if request.method == 'POST':
		rep = getRep(request.data)
		return json.dumps({"rep": rep})
	else:
		return "Error!"

@app.route('/ppdbAll', methods=['GET', 'POST'])
def ppdbAll():
	if request.method == 'POST':
		phr_list = request.get_json()['data']
		rep_list = []
		for phr in phr_list: rep_list.append(getRep(phr))
		return json.dumps({"data": rep_list})
	else:
		return 'Error! in ppdb'

if __name__ == "__main__":
	print("PPDB Server Running")
	app.run(host='0.0.0.0', port=args.port)