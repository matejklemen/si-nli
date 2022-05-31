import pandas as pd
from collections import defaultdict
import random

clusters = defaultdict(list)
with open('clusters-filtered_all-LaBSE-eps_0.3.txt') as cluster:
    cluster_id = 0
    for line in cluster:
        if line.startswith('Cluster#'):
            cluster_id = int(line.strip().split('#')[1][:-1])
        elif '-' in line:
            sentence = line.strip()[2:]
            clusters[cluster_id].append(sentence)
        else:
            continue

final = defaultdict(list)
for cluster_id, candidates in clusters.items():
    random.shuffle(candidates)
    final['source'].append(candidates[0])
    final['target'].append(candidates[1])
    final['id'].append(cluster_id)
final = pd.DataFrame(final)
final.to_csv('slo-nli-test_set.csv', index=False)

