from typing import List, Union

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class TransformersEmbedding:
	def __init__(self, pretrained_name_or_path: str, max_length: int, batch_size: int = 32,
				 device: Union["cuda", "cpu"] = "cuda"):
		assert device in ["cuda", "cpu"]
		if device == "cuda" and not torch.cuda.is_available():
			raise ValueError("Selected 'cuda' device but no CUDA-capable device found. To fix this, set device='cpu'")
		self.device_str = device
		self.device = torch.device(device)

		self.model = AutoModel.from_pretrained(pretrained_name_or_path, output_hidden_states=True).to(self.device)
		self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name_or_path)

		self.batch_size = batch_size
		self.max_length = max_length

	@torch.no_grad()
	def embed_sentences(self, sentences: List[str], use_layer: int = -1,
						aggregation: Union["mean", "cls"] = "mean"):
		encoded_input = self.tokenizer(sentences, max_length=self.max_length, padding="max_length", truncation=True,
									   return_tensors="pt")

		# aggregation function accepts [num_valid_tokens, hidden_size] tensor
		if aggregation == "mean":
			agg_func = lambda t: torch.mean(t, dim=0)
		elif aggregation == "cls":
			agg_func = lambda t: t[0]
		else:
			raise NotImplementedError

		representations = []
		num_batches = (len(sentences) + self.batch_size - 1) // self.batch_size
		for idx_batch in tqdm(range(num_batches), total=num_batches):
			s_b, e_b = idx_batch * self.batch_size, (idx_batch + 1) * self.batch_size
			batch_on_device = {_k: _v[s_b: e_b].to(self.device) for _k, _v in encoded_input.items()}

			res = self.model(**batch_on_device).hidden_states

			relevant_res = res[use_layer]  # [num_examples, max_length, hidden_size]
			for _idx_in_batch in range(relevant_res.shape[0]):
				curr_attn = batch_on_device["attention_mask"][_idx_in_batch].bool()
				representations.append(agg_func(relevant_res[_idx_in_batch][curr_attn]).cpu())

		representations = torch.stack(representations)
		return representations


if __name__ == "__main__":
	sample_sentence = "Zračnoprevozna divizija je divizija, ki za transport in/ali bojevanje uporablja jadralna letala oz. helikopterje."

	embedder = TransformersEmbedding("EMBEDDIA/sloberta",
									 max_length=50,
									 device="cpu")
	embedded_sent = embedder.embed_sentences([sample_sentence])
	print(embedded_sent.shape)



