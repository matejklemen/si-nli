import re
import string
from tqdm import tqdm


def clean_sentence(sent: str):
	sent_prepr = sent

	# If the sentence does not end with a punctuation, end it with "."
	if not any(sent_prepr.endswith(curr_punct) for curr_punct in string.punctuation):
		sent_prepr = f"{sent_prepr}."

	sent_prepr = re.sub(r"\.{2,}", ".", sent_prepr)
	sent_prepr = re.sub(r"(»|«)", "\"", sent_prepr)

	return sent_prepr


def filter_sentences(sentences: list):
	import classla
	# filter candidate sentences based on rules
	nlp = classla.Pipeline('sl', processors='tokenize,pos', use_gpu=True)
	candidates = []
	for sentence in tqdm(sentences):
		# annotate
		doc = nlp(sentence)
		# check if sent is appropriate
		filter_tags = {'NOUN', 'DET', 'PROPN', 'PRON'}
		sentence_tags = set([w.upos for w in doc.sentences[0].words])
		if 'VERB' in sentence_tags or 'AUX' in sentence_tags:
			for tag in filter_tags:
				if tag in sentence_tags:
					candidates.append(sentence)
					break
	return candidates

