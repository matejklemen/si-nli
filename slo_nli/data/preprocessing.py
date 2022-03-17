import re
import string


def clean_sentence(sent: str):
	sent_prepr = sent

	# If the sentence does not end with a punctuation, end it with "."
	if not any(sent_prepr.endswith(curr_punct) for curr_punct in string.punctuation):
		sent_prepr = f"{sent_prepr}."

	sent_prepr = re.sub(r"\.{2,}", ".", sent_prepr)
	sent_prepr = re.sub(r"(»|«)", "\"", sent_prepr)

	return sent_prepr
