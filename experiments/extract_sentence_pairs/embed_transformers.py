import argparse
from itertools import chain
from multiprocessing import Pool, set_start_method

from time import time
from typing import List

import classla
import numpy as np
import pandas as pd
import torch

from slo_nli.data.data_loader import load_cckres
from slo_nli.data.preprocessing import clean_sentence
from slo_nli.features.extract_features import TransformersEmbedding

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, help="Path to cckres (.vert)",
					default="/home/matej/Documents/slo_nli/data/raw/cckres.vert")
parser.add_argument("--target_path", type=str, default="embeddings.csv")

parser.add_argument("--embedding_type", type=str, choices=["word", "sentence"], default="word")
parser.add_argument("--pretrained_name_or_path", type=str, default="EMBEDDIA/sloberta")
parser.add_argument("--layer_id", type=int, default=-1, help="Hidden layer to use as token embeddings. For example, "
															 "-1 = last layer.")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_length", type=int, default=128, help="Max length of sequences used in transformers")

parser.add_argument("--num_stanza_processes", type=int, default=1)
parser.add_argument("--use_cpu", action="store_true")


def classla_upos(pipe_data_tup) -> List[set]:
	pipe, curr_sents = pipe_data_tup
	curr_tags = []
	for idx_sent in range(len(curr_sents)):
		doc = pipe(curr_sents[idx_sent])
		tags = set([w.upos for w in doc.sentences[0].words])
		curr_tags.append(tags)

	return curr_tags


if __name__ == "__main__":
	set_start_method("spawn")
	args = parser.parse_args()
	if not torch.cuda.is_available():
		args.use_cpu = True
		print("Warning: implicitly set '--use_cpu' because no CUDA-capable device could be found")

	data = load_cckres(args.data_path, preprocess_func=clean_sentence)
	print(f"Loaded dataset with {data.shape[0]} examples")

	min_tokens, max_tokens = 10, 40
	data = data.loc[np.logical_and(data["num_tokens"] > 10, data["num_tokens"] < 40)].reset_index(drop=True)
	print(f"After length filtering: {data.shape[0]} examples")

	nlp = classla.Pipeline('sl', processors='tokenize,pos', use_gpu=True, tokenize_no_ssplit=True)

	ts = time()
	num_processes = args.num_stanza_processes
	data_chunks = [data.iloc[curr_indices]["sentence"].tolist()
				   for curr_indices in np.array_split(np.arange(data.shape[0]), num_processes)]
	with Pool(num_processes) as pool:
		upos_tags = list(chain(*pool.map(classla_upos, [(nlp, _curr_data) for _curr_data in data_chunks])))
	print(f"Classla tagging took {time() - ts:.3f}s (num_processes={num_processes})")

	# Filter sentences with "odd" structure
	mask_valid_sentence = np.zeros(data.shape[0], dtype=bool)
	for idx_ex in range(data.shape[0]):
		curr_ex = data.iloc[idx_ex]
		curr_upos = upos_tags[idx_ex]

		mask_valid_sentence[idx_ex] = ("VERB" in curr_upos or "AUX" in curr_upos) and \
									  len(curr_upos & {'NOUN', 'DET', 'PROPN', 'PRON'}) > 0

	data = data.loc[mask_valid_sentence].reset_index(drop=True)
	print(f"After UPOS filtering: {data.shape[0]} examples")

	ts = time()
	if args.embedding_type == "word":
		embedder = TransformersEmbedding(pretrained_name_or_path=args.pretrained_name_or_path,
										 max_length=args.max_length, batch_size=args.batch_size,
										 device=("cpu" if args.use_cpu else "cuda"))

		sent_reprs = embedder.embed_sentences(data["sentence"].tolist(), use_layer=args.layer_id).numpy()
	elif args.embedding_type == "sentence":
		from sentence_transformers import SentenceTransformer
		model_name = 'LaBSE'
		model = SentenceTransformer(model_name)
		sent_reprs = model.encode(data["sentence"].tolist(),
								  show_progress_bar=True, convert_to_numpy=True, batch_size=args.batch_size)
	else:
		raise NotImplementedError(f"--embedding_type='{args.embedding_type}' not supported")

	te = time()

	data = pd.concat((data.reset_index(drop=True),
					  pd.DataFrame(sent_reprs, columns=[f"x_{_idx_feat}" for _idx_feat in range(sent_reprs.shape[1])])),
					 axis=1)

	print(f"Embedding took {te - ts:.3f}s... Writing {data.shape[0]} examples")
	data.to_csv(args.target_path, index=False, sep=",")
