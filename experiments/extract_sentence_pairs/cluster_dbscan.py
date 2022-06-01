import argparse

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument("--embeddings_path", type=str, default="embeddings_sloberta_last_mean.csv")
parser.add_argument("--sample_size", type=int, default=None)
parser.add_argument("--pca_components", type=int, default=100)
parser.add_argument("--dbscan_eps", type=float, default=0.3)
parser.add_argument("--dbscan_minsamples", type=int, default=2)

parser.add_argument("--target_path", type=str, default="cckres_candidates.csv")

if __name__ == "__main__":
	args = parser.parse_args()
	data = pd.read_csv(args.embeddings_path, sep=",")
	if args.sample_size is not None:
		data = data.sample(n=int(args.sample_size))
	embeddings = data.iloc[:, 4:].values

	preprojection = PCA(n_components=args.pca_components).fit_transform(embeddings)

	clusters = DBSCAN(metric="cosine", eps=args.dbscan_eps, min_samples=args.dbscan_minsamples,
					  n_jobs=4).fit_predict(preprojection)
	grouped = {}
	for _i, cl in enumerate(clusters):
		if cl != -1:
			current_cl = grouped.get(cl, [])
			current_cl.append(data.iloc[_i]["sentence"])
			grouped[cl] = current_cl

	candidates = {
		"cluster_id": [],
		"hypothesis": [],
		"premise": []
	}
	for cluster_id, items in grouped.items():
		if len(items) < 2:
			continue

		shuf_indices = np.random.permutation(len(items))
		candidates["cluster_id"].append(cluster_id)
		candidates["hypothesis"].append(items[shuf_indices[0]])
		candidates["premise"].append(items[shuf_indices[1]])

	candidates = pd.DataFrame(candidates)
	print(f"Writing {candidates.shape[0]} candidates to '{args.target_path}'")
	candidates.to_csv(args.target_path, sep=",", index=False)

	# TODO: write clusters in its original form (for evaluation?)
	# with open(f"clusters.txt", "w", encoding="utf-8") as f:
	# 	for cl, items in grouped.items():
	# 		print(f"Cluster#{cl}:", file=f)
	# 		for ex in items:
	# 			print(f"\t- {ex}", file=f)
	# 		print("", file=f)


