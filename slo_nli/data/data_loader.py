from typing import Optional, Callable

import pandas as pd
from bs4 import BeautifulSoup
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
			tokens = [line.split("\t")[0] for line in curr_sent.text.strip().split("\n")]

			sent_text = eff_prepr_func(" ".join(tokens))
			if dedup and (sent_text in cached_sents):
				continue

			loaded_data["doc_id"].append(doc_id)
			loaded_data["genre"].append(genre)
			loaded_data["sentence"].append(sent_text)
			loaded_data["num_tokens"].append(len(tokens))
			cached_sents.add(sent_text)

	return pd.DataFrame(loaded_data)




