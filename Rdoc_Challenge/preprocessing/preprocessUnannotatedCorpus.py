'''
Created on 8 July 2016

@author: madhumita

Pre-processes the unannotated corpus.
Replaces the space between consecutive terms in the corpus that form a UMLS concept (e.g., "stomach ache"), by an underscore (e.g. 'stomach_ache').
Writes the new corpus to another file on disk.
It takes input of a tokenized corpus where each token is lower cased and separated by a space.
'''

from preprocessing.DictionaryMatcher import DictionaryMatcher
from collections import defaultdict
from glob import iglob
import utils
import json

PATH_CORPUS_IN = '../../resources/corpus/unannotated/all_processed.txt'
PATH_CORPUS_OUT = '../../resources/corpus/unannotated/'
processedCorpusName = "all_processed_concepts.txt"

def identify_and_replace_concepts(loadLibrary, f_in, f_out):

    f_corpus = open(f_in)
    f_corpus_out = open(f_out,'w')
    
    matcher = DictionaryMatcher(4, 0.80)
    if loadLibrary:
        matcher.loadLibrary('FILEUMLS')
        matcher.saveModel()
    else:
        matcher.loadModel('FILEUMLS')
    
    for i,line in enumerate(f_corpus):
        concepts = matcher.process_text(text=line, use_chunks=False, tokenized=False)
        for concept in concepts:
            concept_str = ' '.join(concept[1][1][2])
    #         utils.out("Obtained concept,")
    #         utils.out(concept_str)
            line = line.replace(concept_str,concept_str.replace(' ','_'))
        if i and i%1000 == 0:
            utils.out("Processed "+str(i)+" lines")
        f_corpus_out.write(line)


def identify_concepts(lines, loadlibrary):

    matcher = DictionaryMatcher(4, 0.80)
    if loadlibrary:
        matcher.loadLibrary('DSM')
        matcher.saveModel()
    else:
        matcher.loadModel('FILEUMLS')

    all_concepts = []

    for line in lines:

        line = line.lower().replace("<newline>", " ")

        concepts = matcher.process_text(line, use_chunks=False, tokenized=False)
        all_concepts.extend(concepts)

    d = defaultdict(list)

    for k, concept in all_concepts:
        d[" ".join(k)].append(concept)

    return d

if __name__ == '__main__':
    # identify_and_replace_concepts(loadLibrary=True, f_in = PATH_CORPUS_IN, f_out=PATH_CORPUS_OUT+processedCorpusName)
    all_conc = identify_concepts(open("../input/concatenated/annotated.txt").readlines(), loadlibrary=True)

    new = {}

    for k, v in all_conc.items():
        if not v:
            continue
        try:
            new[k] = [[x[0], [x[1][0], x[1][1], list(x[1][2])]] for x in v]
        except IndexError:
            print(v)

    json.dump(new, open("all_concepts_dsm.json"))
