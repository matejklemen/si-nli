import pandas as pd
import classla
import stanza
from collections import Counter

from tqdm import tqdm


def validate_sentence(s):
    # Vprašanja – če se poved konča z vprašajem, se jo izloči
    # Podpičja, dvopičja – če se poved konča s podpičjem ali dvopičjem, se jo izloči.
    if s[-1] in {'?', ';', ':'}:
        return False

    # annotate
    doc = nlp(s)
    assert len(doc.sentences) == 1  # verify that there is one sentence

    # get tags
    upos = [[w.upos for w in sent.words] for sent in doc.sentences][0]
    upos_counter = Counter(upos)

    xpos = [[w.xpos for w in sent.words] for sent in doc.sentences][0]

    # Če poved ne vsebuje nobenega glagola, se jo izloči.
    if 'VERB' not in upos and 'AUX' not in upos:
        return False

    # Velelnik – če poved vsebuje katerokoli oblikoskladenjsko oznako (MTE-6) za velelnik, se jo izloči.
    velelnik = ('Vmem', 'Vmpm', 'Vmbm', 'Va-m')  # msd startswith
    for msd in xpos:
        if msd.startswith(velelnik):
            return False

    # Če poved vsebuje samo nedoločnik (in nobenega drugega glagola), se jo izloči.
    nedolocnik = {'Vmen', 'Vmpn', 'Vmbn', 'Va-n'}
    for n in nedolocnik:
        if upos_counter['VERB'] + upos_counter['AUX'] == 1 and n in xpos:
            return False

    deleznik = ('Vmep', 'Vmpp', 'Vmbp', 'Va-p')  # msd startswith
    for msd in xpos:
        if msd.startswith(deleznik):
            # Premi govor – če poved vsebuje vsaj dva narekovaja in deležnik, se jo izloči.
            char_counter = Counter(s)
            if char_counter["\""] > 1 or char_counter["\'"] > 1:  # check for single or double quotes
                    return False

            # Če poved vsebuje samo deležnik (in nobenega drugega glagola), se jo izloči.
            if upos_counter['VERB'] + upos_counter['AUX'] == 1:
                return False

    # sentence is OK
    return True


if __name__ == '__main__':
    nlp = stanza.Pipeline('sl', processors='tokenize,pos', use_gpu=True, tokenize_no_ssplit=True)

    parlamint = '/storage/public/slo-nli-wip/parlamint_candidates_auto_annotated.csv'
    cckres = '/storage/public/slo-nli-wip/cckres_candidates_auto_annotated.csv'

    df = pd.read_csv(parlamint)
    df = pd.concat([df, pd.read_csv(cckres)])

    filtered = []
    for h, p in tqdm(zip(df['hypothesis'], df['premise'])):
        if validate_sentence(h) and validate_sentence(p):
            filtered.append([h, p])

    filtered = pd.DataFrame(filtered, columns=['hypothesis', 'premise'])
    df.to_csv('filtered-sentence-pairs.csv', index=False)