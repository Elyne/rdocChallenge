'''
This file creates the allUmls.txt, dsm4.txt, and dsm4_rel1.txt files
This file can also create lexicons based on strings occurring in UMLS.

Created on 8 Jun 2016

@author: elyne
'''

import config as cfg
from utilIterators import UMLSTableIterator, FileIterator
from crawlers.PubMedCrawler import PubMedCrawler
from crawlers.WikipediaCrawler import WikipediaCrawler

def createInitialLexicons():
    
    ## First, we create a lexicon based on all UMLS concepts
    fout = open(cfg.PATH_RESOURCES + "allUmls.txt",'w')
    cnt = 0
    for line in UMLSTableIterator("SELECT CUI, STR FROM MRCONSO WHERE LAT='ENG'"):
        fout.write(line[0] + "-" + str(cnt) + ";" + line[1].replace(';',',') + "\n")
        cnt+=1
    fout.close()
    
    ## Then, we create a lexicon based on DSM-IV concepts
    fout = open(cfg.PATH_RESOURCES + "dsm4.txt",'w')
    cnt = 0
    for line in UMLSTableIterator("SELECT CUI, STR FROM MRCONSO where SAB='DSM4'"):
        for rec in UMLSTableIterator("SELECT CUI, STR FROM MRCONSO where CUI='"+line[0]+"' AND LAT='ENG'"):
            fout.write(rec[0] + "-" + str(cnt) + ";" + rec[1].replace(';',',') + "\n")
            cnt += 1
        cnt+=1
    fout.close()
    
    ## Then, a lexicon based on DSM-IV concepts and all direct relations
    ##Make list of initial cuis
    x = dict()
    for line in UMLSTableIterator("SELECT CUI, STR FROM MRCONSO where SAB='DSM4'"):
        x[line[0]] = 0
        for line2 in UMLSTableIterator("SELECT CUI2, RELA FROM MRREL WHERE CUI1='"+line[0]+"'"):
            if not (line2[0] in x):
                x[line2[0]] = 1
    
    fout = open(cfg.PATH_RESOURCES + "dsm4_rel1.txt",'w')
    cnt = 0
    for cui in x:
        for rec in UMLSTableIterator("SELECT CUI, STR FROM MRCONSO where CUI='"+cui+"' AND LAT='ENG'"):
            fout.write(rec[0] + "-" + str(cnt) + ";" + rec[1].replace(';',',') + "\n")
            cnt += 1
    fout.close()
    
    ## Then, a lexicon based on DSM-IV concepts and all direct relations +2
    ##Make list of initial cuis
    x = dict()
    for line in UMLSTableIterator("SELECT CUI, STR FROM MRCONSO where SAB='DSM4'"):
        x[line[0]] = 0
        for line2 in UMLSTableIterator("SELECT CUI2, RELA FROM MRREL WHERE CUI1='"+line[0]+"'"):
            for line3 in UMLSTableIterator("SELECT CUI2, RELA FROM MRREL WHERE CUI1='"+line2[0]+"'"):
                if not (line3[0] in x):
                    x[line3[0]] = 1
    
    fout = open(cfg.PATH_RESOURCES + "dsm4_rel2.txt",'w')
    cnt = 0
    for cui in x:
        for rec in UMLSTableIterator("SELECT CUI, STR FROM MRCONSO where CUI='"+cui+"' AND LAT='ENG'"):
            fout.write(rec[0] + "-" + str(cnt) + ";" + rec[1].replace(';',',') + "\n")
            cnt += 1
    fout.close()
    
    
    
def perform(keywords):
    ## initialising objects for fetching abstracts/wiki/..
    pubmed = PubMedCrawler()
    wiki = WikipediaCrawler()
    ##sending list of keywords to query instances
    for keyword in keywords:
        pubmed.fetchArticleIdsForTerm(keyword)
        wiki.fetchArticleIdsForTerm(keyword)
        
    wiki.writeQueueToFile()
    pubmed.writeQueueToFile()
        
    wiki.execute()
    pubmed.execute()
    
def createForDSM():
    createInitialLexicons()
    
    ##Reading in keywords from file
    keywords = set()
    for line in FileIterator(cfg.PATH_RESOURCES+'dsm4.txt'):
        #add each addition of additional params, but for the first concept first
        for i in range(0,len(line[1])):
            key = ' '.join(line[1][0:i+1]).lower()
            while '  ' in key:
                key = key.replace('  ',' ')
            keywords.add(key)
        
    keywords = list(keywords)
    perform(keywords)
    
def createForRdoc():
    keywords = set()
    for line in open(cfg.PATH_RESOURCES+'rdoc_terms.txt'):
        #add each addition of additional params, but for the first concept first
        line = line.replace('\n','').strip()
        if (len(line) > 3):
            keywords.add(line)
    perform(keywords)
    
if __name__=='__main__':
    createInitialLexicons()
    #createForDSM()
    #createForRdoc()