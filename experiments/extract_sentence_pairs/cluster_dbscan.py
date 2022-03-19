import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# TODO: This is just starter clustering code, not generalized yet
data = pd.read_csv("embeddings_sloberta_last_mean.csv", sep=",").sample(n=10_000)
embeddings = data.iloc[:, 4:].values

preprojection = PCA(n_components=100).fit_transform(embeddings)

clusters = DBSCAN(metric="cosine", eps=0.3, min_samples=2).fit_predict(preprojection)
grouped = {}
for _i, cl in enumerate(clusters):
	if cl != -1:
		current_cl = grouped.get(cl, [])
		current_cl.append(data.iloc[_i]["sentence"])
		grouped[cl] = current_cl

with open(f"clusters.txt", "w", encoding="utf-8") as f:
	for cl, items in grouped.items():
		print(f"Cluster#{cl}:", file=f)
		for ex in items:
			print(f"\t- {ex}", file=f)
		print("", file=f)


