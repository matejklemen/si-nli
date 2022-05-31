import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import time

# import sentences
with open('sentences-filtered.txt', 'r') as inp_file:
    candidates = [sent.strip() for sent in inp_file]

# # calculate embeddings
model_name = 'LaBSE'
model = SentenceTransformer(model_name)
embeddings = model.encode(candidates, show_progress_bar=True, convert_to_numpy=True, batch_size=1024)
# np.save('embeddings-cckres.npy', embeddings)
# print('Embeddings saved on disk.')

# load embeddings
embeddings = np.load('embeddings-cckres.npy')
print(f'Num of embeddings {len(embeddings)}')

# reduce dimensions of embeddings
print('Start PCA ...')
start_time = time.time()
preprojection = PCA(n_components=128).fit_transform(embeddings)
print("PCA finished in --- %s seconds ---" % (time.time() - start_time))

# cluster and write to disk
eps = 0.3
min_samples = 5
print('Start DBSCAN ...')
start_time = time.time()
clusters = DBSCAN(metric="cosine", eps=eps, min_samples=min_samples).fit_predict(preprojection)
print("DBSCAN finished in --- %s seconds ---" % (time.time() - start_time))

grouped = {}
for _i, cl in enumerate(clusters):
    if cl != -1:
        current_cl = grouped.get(cl, [])
        current_cl.append(candidates[_i])
        grouped[cl] = current_cl

with open(f"clusters-filtered_all-{model_name}-eps_{eps}-minsamples_{min_samples}.txt", "w", encoding="utf-8") as f:
    for cl, items in grouped.items():
        print(f"Cluster#{cl}:", file=f)
        for ex in items:
            print(f"\t- {ex}", file=f)
        print("", file=f)
