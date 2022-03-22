import argparse
import logging
import os
import sys

import pandas as pd
import scipy.stats
import torch.cuda
from torch import cosine_similarity

try:
	from sentence_transformers import SentenceTransformer
except:
	raise ImportError(f"run `pip install sentence-transformers`")

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_dir", type=str, default="debug")
parser.add_argument("--train_path",
					default="/home/matej/Documents/paraphrase-nli/experiments/STSB_UTIL/v0-machine-translated/translated.SL.train.tsv")
parser.add_argument("--dev_path",
					default="/home/matej/Documents/paraphrase-nli/experiments/STSB_UTIL/v0-machine-translated/translated.SL.dev.tsv")
parser.add_argument("--representation", choices=["sentence", "token"],
					default="sentence")
parser.add_argument("--pretrained_name_or_path", type=str,
					default="distiluse-base-multilingual-cased-v2")
parser.add_argument("--batch_size", type=int,
					default=16)

parser.add_argument("--use_cpu", action="store_true")

if __name__ == "__main__":
	args = parser.parse_args()
	if not os.path.exists(args.experiment_dir):
		os.makedirs(args.experiment_dir)

	device_str = "cpu" if args.use_cpu else "cuda"
	if not torch.cuda.is_available():
		args.use_cpu, device_str = True, "cpu"
		logging.warning("Warning: implicitly set '--use_cpu' because no CUDA-capable device could be found")

	# Set up logging to file and stdout
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	for curr_handler in [logging.StreamHandler(sys.stdout),
						 logging.FileHandler(os.path.join(args.experiment_dir, "experiment.log"))]:
		curr_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s"))
		logger.addHandler(curr_handler)

	test_df = pd.concat((
		pd.read_csv(args.train_path, sep="\t"),
		pd.read_csv(args.dev_path, sep="\t")
	)).reset_index(drop=True)
	seq1, seq2 = test_df["translated.sentence1"].tolist(), test_df["translated.sentence2"].tolist()
	gt_scores = test_df["score"].values
	logging.info(f"Loaded {test_df.shape[0]} examples...")

	model = SentenceTransformer(args.pretrained_name_or_path, device=device_str)
	# Obtain a representation for first and second sequences in pairs
	emb_seq1 = model.encode(seq1, batch_size=args.batch_size, device=device_str, convert_to_tensor=True,
							show_progress_bar=True)
	emb_seq2 = model.encode(seq2, batch_size=args.batch_size, device=device_str, convert_to_tensor=True,
							show_progress_bar=True)

	similarities = cosine_similarity(emb_seq1, emb_seq2).cpu().numpy()

	pearson_corr, _ = scipy.stats.pearsonr(similarities, gt_scores)
	spearman_corr, _ = scipy.stats.spearmanr(similarities, gt_scores)

	test_df["similarity"] = similarities
	test_df.to_csv(os.path.join(args.experiment_dir, "test_similarities.tsv"), sep="\t", index=False)

	logging.info(f"[Results] PearsonCorr = {pearson_corr:.3f}, SpearmanCorr = {spearman_corr:.3f},"
				 f"avg = {(pearson_corr + spearman_corr) / 2:.3f}")
