import re
import string

import pandas as pd
import classla
# import stanza
from collections import Counter

from nltk import word_tokenize
from tqdm import tqdm

from lemmagen3 import Lemmatizer


def validate_sentence(s):
    # Vprašanja – če se poved konča z vprašajem, se jo izloči
    # Podpičja, dvopičja – če se poved konča s podpičjem ali dvopičjem, se jo izloči.
    if s[-1] in {'?', ';', ':'}:
        return False

    # annotate
    doc = nlp(s)

    # check if there is more than one sentence
    if len(doc.sentences) > 1:  # verify that there is one sentence
        return False

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
            # # Premi govor – če poved vsebuje vsaj dva narekovaja in deležnik, se jo izloči.
            # char_counter = Counter(s)
            # if char_counter["\""] > 1 or char_counter["\'"] > 1:  # check for single or double quotes
            #         return False

            # Če poved vsebuje samo deležnik (in nobenega drugega glagola), se jo izloči.
            if upos_counter['VERB'] + upos_counter['AUX'] == 1:
                return False

    # sentence is OK
    return True


def lemmatize(text):
    content_lemmatized = []
    for token in word_tokenize(text, language='slovene'):
        lemma = lem_sl.lemmatize(token)
        content_lemmatized.append(lemma.lower())
    return content_lemmatized


def get_entities(text):
    entities = []
    annotated_text = nlp(text)
    for sent in annotated_text.sentences:
        for ent in sent.entities:
            entity = ""
            for t in ent.tokens:
                if not entity:
                    entity = t.text
                else:
                    entity += " " + t.text
            entities.append(entity)
    return entities


def validate_pair(h, p, check_numbers=True, check_entities=True):
    if check_numbers:
        numbers_h, numbers_p = re.findall('\d+', h), re.findall('\d+', p)
        for num in numbers_h:
            if num not in numbers_p:
                return False

    if check_entities:
        entities_h, entities_p = get_entities(h), get_entities(p)

        lemma_entities_h = []
        for entity in entities_h:
            lemmatized_entity = ""
            for token in word_tokenize(entity, language='slovene'):
                if not lemmatized_entity:
                    lemmatized_entity = token
                else:
                    lemmatized_entity += " " + lem_sl.lemmatize(token)
            lemma_entities_h.append(lemmatized_entity)

        lemma_entities_p = []
        for entity in entities_p:
            lemmatized_entity = ""
            for token in word_tokenize(entity, language='slovene'):
                if not lemmatized_entity:
                    lemmatized_entity = token
                else:
                    lemmatized_entity += " " + lem_sl.lemmatize(token)
            lemma_entities_p.append(lemmatized_entity)

        diff = 0
        for ent in lemma_entities_h:
            if ent not in lemma_entities_p:
                diff += 1
        if diff > 1:
            return False

    return True


if __name__ == '__main__':
    nlp = classla.Pipeline('sl', processors='tokenize,pos,ner', use_gpu=True, tokenize_no_ssplit=True)
    lem_sl = Lemmatizer('sl')

    df = pd.read_csv('/storage/public/slo-nli-wip/cckres_selected_filtered.csv')

    valid = []
    invalid = []
    for h, p in tqdm(zip(df['hypothesis'], df['premise'])):
        if validate_sentence(h) and validate_sentence(p) and validate_pair(h, p):
            valid.append([h, p])
        else:
            invalid.append([h, p])

    valid = pd.DataFrame(valid, columns=['hypothesis', 'premise'])
    valid.to_csv('cckres_selected_valid.csv', index=False)
    
    invalid = pd.DataFrame(invalid, columns=['hypothesis', 'premise'])
    invalid.to_csv('cckres_selected_invalid.csv', index=False)