import os
from typing import Optional, Callable

import pandas as pd
from bs4 import BeautifulSoup
from nltk import sent_tokenize, word_tokenize
from tqdm import tqdm


def load_cckres(path_to_vert: str,
				dedup=True, preprocess_func: Optional[Callable] = None) -> pd.DataFrame:
	with open(path_to_vert, "r", encoding="utf-8") as f:
		soup = BeautifulSoup(f, features="lxml")

	eff_prepr_func = (lambda o: o) if preprocess_func is None else preprocess_func
	cached_sents = set()

	loaded_data = {
		"doc_id": [],
		"genre": [],
		"sentence": [],
		"num_tokens": [],  # in case length-based filtering is needed afterwards
	}
	all_docs = soup.find_all("text")
	for curr_doc in tqdm(all_docs):
		doc_id = curr_doc["id"]
		# e.g. extract "časopis" from class="tisk/periodično/časopis"
		genre = curr_doc["class"][0].split("/")[-1]

		sents = curr_doc.find_all("s")
		for curr_sent in sents:
			tokens = []
			for token_info in curr_sent.text.strip().split("\n"):
				parts = token_info.split("\t")
				tokens.append(parts[0])

			sent_text = eff_prepr_func(" ".join(tokens))
			if dedup and (sent_text in cached_sents):
				continue

			loaded_data["doc_id"].append(doc_id)
			loaded_data["genre"].append(genre)
			loaded_data["sentence"].append(sent_text)
			loaded_data["num_tokens"].append(len(tokens))
			cached_sents.add(sent_text)

	return pd.DataFrame(loaded_data)


def load_parlamint(parlamint_dir: str,
				   dedup=True, preprocess_func: Optional[Callable] = None) -> pd.DataFrame:

	fnames = [curr_fname for curr_fname in os.listdir(parlamint_dir)
			  if os.path.isfile(os.path.join(parlamint_dir, curr_fname)) and
			  curr_fname.endswith(".txt") and
			  "README" not in curr_fname]

	eff_prepr_func = (lambda o: o) if preprocess_func is None else preprocess_func
	cached_sents = set()

	all_data = {
		"utterance_id": [],
		"sentence": [],
		"num_tokens": []
	}
	for curr_fname in tqdm(fnames):
		curr_path = os.path.join(parlamint_dir, curr_fname)
		with open(curr_path, "r", encoding="utf-8") as f:
			for curr_line in f:
				u_id, utterance = curr_line.strip().split("\t")

				for idx_sent, curr_sent in enumerate(sent_tokenize(utterance, language="slovene")):
					sent_text = eff_prepr_func(curr_sent)
					if dedup and (sent_text in cached_sents):
						continue

					all_data["utterance_id"].append(f"{u_id}_s{idx_sent}")
					all_data["sentence"].append(curr_sent)
					all_data["num_tokens"].append(len(word_tokenize(curr_sent)))
					cached_sents.add(curr_sent)

	return pd.DataFrame(all_data)



