import argparse
import os
from typing import List, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="/home/matej/Downloads/slo-nli_test_set/slo-nli-test_set.csv")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--mcd_iters", type=int, default=0)


# TODO: pass pairs through NLI model and find if there are certain ones
def filter_pairs(sentence_pairs: List[Tuple[str, str]], pretrained_name_or_path, batch_size=8, mcd_iters=0):
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	tokenizer = AutoTokenizer.from_pretrained(pretrained_name_or_path)
	model = AutoModelForSequenceClassification.from_pretrained(pretrained_name_or_path).to(device)

	assert mcd_iters >= 0
	if mcd_iters > 0:
		num_iters = int(mcd_iters)
		model.train()
	else:
		num_iters = 1
		model.eval()

	encoded = tokenizer(sentence_pairs, max_length=256, padding="max_length", truncation=True, return_tensors="pt")
	num_batches = (encoded["input_ids"].shape[0] + batch_size - 1) // batch_size

	with torch.no_grad():
		pred_probas = []
		for idx_iter in range(num_iters):
			curr_probas = []
			for idx_b in tqdm(range(num_batches), total=num_batches):
				s_b, e_b = idx_b * batch_size, (idx_b + 1) * batch_size
				input_data = {k: v[s_b: e_b].to(device) for k, v in encoded.items()}

				res = model(**input_data)
				probas = torch.softmax(res["logits"], dim=-1)
				curr_probas.append(probas.cpu())

			curr_probas = torch.cat(curr_probas)
			pred_probas.append(curr_probas)

	pred_probas = torch.stack(pred_probas)
	mean_probas = torch.mean(pred_probas, dim=0)  # [num_examples, num_labels]
	sd_probas = torch.zeros_like(mean_probas)
	if num_iters > 1:
		sd_probas = torch.std(pred_probas, dim=0)

	# TODO: find very certain examples
	argmax_preds = torch.argmax(mean_probas, dim=-1)
	argmax_mean_probas = mean_probas[torch.arange(argmax_preds.shape[0]), argmax_preds]
	argmax_sd_probas = sd_probas[torch.arange(argmax_preds.shape[0]), argmax_preds]

	sort_indices = torch.argsort(argmax_mean_probas, descending=True).tolist()

	return {
		"input_pairs": [sentence_pairs[_i] for _i in sort_indices],
		"preds": argmax_preds[sort_indices].tolist(),
		"mean_probas": argmax_mean_probas[sort_indices].tolist(),
		"sd_probas": argmax_sd_probas[sort_indices].tolist()
	}


# TODO: IDX2LABEL is model-handle specific!
if __name__ == "__main__":
	args = parser.parse_args()

	model_handle = "vicgalle/xlm-roberta-large-xnli-anli"
	IDX2LABEL = {0: "contradiction", 1: "neutral", 2: "entailment"}

	# model_handle = "symanto/xlm-roberta-base-snli-mnli-anli-xnli"
	# IDX2LABEL = {0: "entailment", 1: "neutral", 2: "contradiction"}

	# pair = ("Ista številka v omenjenih tabelah in slikah tako zadeva isto šolo .",
	# 		"Številke na levi strani slike so enake številkam šol v tabelah tega poglavja .")
	# input_pairs = [pair]

	data_path = args.data_path
	fname = "".join(args.data_path.split(os.path.sep)[-1].split(".")[:-1])
	data = pd.read_csv(data_path, sep=",")
	input_pairs = list(zip(data["premise"].tolist(), data["hypothesis"].tolist()))

	res = filter_pairs(input_pairs, pretrained_name_or_path=model_handle,
					   batch_size=args.batch_size, mcd_iters=args.mcd_iters)
	num_examples = len(res["input_pairs"])

	pd.DataFrame(res).to_csv(f"{fname}_auto_annotated.csv", sep=",", index=False)

	for idx_ex in range(num_examples):
		curr_pair = res["input_pairs"][idx_ex]
		pred = res["preds"][idx_ex]
		mean_proba = res["mean_probas"][idx_ex]
		sd_proba = res["sd_probas"][idx_ex]

		print(f"(pred={pred}, p = {mean_proba:.3f} ({sd_proba:.3f})) {curr_pair}")
