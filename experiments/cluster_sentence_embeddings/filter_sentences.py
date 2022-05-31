import os

import classla
import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Only consider sentences that are between min_sent_len and max_sent_len characters long
min_sent_len = 30
max_sent_len = 200

# Input files. We interpret every line as sentence.
source_file = "/home/azagar/myfiles/slo-nli/data/processed/sentences.txt"
print("Read source file")
source_sentences = []
with open(source_file, 'r') as f:
    for line in tqdm.tqdm(f):
        if min_sent_len <= len(line) <= max_sent_len:
            source_sentences.append(line)

# # select number of senteces
# num_of_candidates = 50000
# source_sentences = source_sentences[:num_of_candidates]

# filter candidate sentences based on rules
nlp = classla.Pipeline('sl', processors='tokenize,pos', use_gpu=True)
candidates = []
non_candidates = []
for sentence in tqdm.tqdm(source_sentences):
    # annotate
    doc = nlp(sentence)

    # check if sent is appropriate
    filter_tags = {'NOUN', 'DET', 'PROPN',
                   'PRON'}  # TODO: use deprels for more accurate general sentence representation
    sentence_tags = set([w.upos for w in doc.sentences[0].words])
    sentence_ok = False
    if 'VERB' in sentence_tags or 'AUX' in sentence_tags:
        for tag in filter_tags:
            if tag in sentence_tags:
                sentence_ok = True
                candidates.append(sentence)
                break
    if not sentence_ok:
        non_candidates.append(sentence)

# save filtered sentences
with open('sentences-filtered.txt', 'w') as out:
    for s in candidates:
        out.write(s)
