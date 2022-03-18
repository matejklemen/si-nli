import argparse

import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer

from slo_nli.data.data_loader import load_cckres
from slo_nli.data.preprocessing import clean_sentence
from slo_nli.features.extract_features import AugmentedFeatureExtractionPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, help="Path to cckres (.vert)",
					default="/home/matej/Documents/slo_nli/data/raw/cckres.vert")
parser.add_argument("--target_path", type=str, default="embeddings.csv")
parser.add_argument("--layer_id", type=int, default=-1, help="Hidden layer to use as token embeddings")
parser.add_argument("--device_id", type=int, help="-1 = CPU, otherwise ID of CUDA-capable device",
					default=-1)

if __name__ == "__main__":
	args = parser.parse_args()
	data = load_cckres(args.data_path, preprocess_func=clean_sentence)

	# Note: use keyword arguments!
	pipe = AugmentedFeatureExtractionPipeline(
		task="feature-extraction", use_layers=[args.layer_id],
		model=AutoModel.from_pretrained("EMBEDDIA/sloberta", output_hidden_states=True),
		tokenizer=AutoTokenizer.from_pretrained("EMBEDDIA/sloberta"),
		device=args.device_id,
		framework="pt")

	# [num_examples, num_layers, num_tokens, hidden_size]
	sent_reprs = pipe(data["sentence"].tolist())
	sent_reprs = np.stack([np.array(curr_emb).mean(axis=0).mean(axis=0) for curr_emb in sent_reprs])

	data = pd.concat((data,
					  pd.DataFrame(sent_reprs, columns=[f"x_{_idx_feat}" for _idx_feat in range(sent_reprs.shape[1])])),
					 axis=1)

	print(f"Writing {data.shape[0]} examples")
	data.to_csv(args.target_path, index=False, sep=",")









