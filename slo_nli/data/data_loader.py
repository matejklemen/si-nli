import os
from logging import warning
from typing import Optional, Callable, Union, Iterable

import pandas as pd
import torch
from bs4 import BeautifulSoup
from nltk import sent_tokenize, word_tokenize
from torch.utils.data import Dataset
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


class TransformersSeqPairDataset(Dataset):
    def __init__(self, **kwargs):
        self.valid_attrs = []
        for attr, values in kwargs.items():
            self.valid_attrs.append(attr)
            setattr(self, attr, values)

        assert len(self.valid_attrs) > 0

    def __getitem__(self, item):
        return {k: getattr(self, k)[item] for k in self.valid_attrs}

    def __len__(self):
        return len(getattr(self, self.valid_attrs[0]))


class SloNLITransformersDataset(TransformersSeqPairDataset):
    def __init__(self, path: Union[str, Iterable[str]], tokenizer,
                 max_length: Optional[int] = None, return_tensors: Optional[str] = None):
        _path = (path,) if isinstance(path, str) else path
        df = pd.concat([pd.read_csv(curr_path, sep="\t") for curr_path in _path]).reset_index(drop=True)

        self.label_names = ["entailment", "neutral", "contradiction"]
        self.label2idx = {curr_label: i for i, curr_label in enumerate(self.label_names)}
        self.idx2label = {i: curr_label for curr_label, i in self.label2idx.items()}

        self.str_premise = df["premise"].tolist()
        self.str_hypothesis = df["hypothesis"].tolist()

        if "label" in df.columns:
            valid_label = list(map(lambda lbl: self.label2idx[lbl], df["label"].tolist()))
        else:
            warning(f"No labels present in file - setting all labels to 0, so you should ignore metrics based on these")
            valid_label = [0] * len(self.str_premise)

        optional_kwargs = {}
        if return_tensors is not None:
            valid_label = torch.tensor(valid_label)
            optional_kwargs["return_tensors"] = "pt"

        if max_length is not None:
            optional_kwargs["max_length"] = max_length
            optional_kwargs["padding"] = "max_length"
            optional_kwargs["truncation"] = "longest_first"

        encoded = tokenizer.batch_encode_plus(list(zip(self.str_premise, self.str_hypothesis)), **optional_kwargs)
        encoded["labels"] = valid_label

        super().__init__(**encoded)

