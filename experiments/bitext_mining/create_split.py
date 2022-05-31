from random import shuffle

with open('sentences.txt', 'r') as f:
	sentences = [line.strip() for line in f]
	shuffle(sentences)

# create test
with open('src.txt', 'w') as src, open('tgt.txt', 'w') as tgt:
	for idx, sent in enumerate(sentences):
		if idx % 1000 == 0:
			src.write(sent)
			src.write('\n')
		else:
			tgt.write(sent)
			tgt.write('\n')

# # subset target
# with open('tgt.txt', 'r') as tgt, open('tgt-very-small.txt', 'w') as tgt_small:
# 	for idx, sent in enumerate(tgt):
# 		if idx > 1000:
# 			break
# 		else:
# 			tgt_small.write(sent)
