'''
Created on 3 Jun 2016

@author: elyne
'''

import config as cfg
import utils
import pickle
import math
import time
from utilIterators import FileIterator, UMLSTableIterator, WindowSlider
from collections import defaultdict, Counter
import re


'''
The UMLS matcher class works by keeping a word-based dictionary, linking to all concepts
It also keeps a 'total idf score' for each concept, so it can decide how well a word is linked to the concept

'''
class DictionaryMatcher():
    def __init__(self, idfThreshold, assignThreshold):
        self.idfs = dict()
        self.tree = dict()
        self.conceptScores = dict()
        self.conceptContent = dict()
        self.threshold = assignThreshold
        self.name = None

        self.retentionThreshold = idfThreshold

    def saveModel(self):
        pickle.dump(self, open(cfg.PATH_TEMP+"dict_"+self.name+".p", "wb"))

    def loadModel(self, name):
        return pickle.load(open(cfg.PATH_TEMP+"dict_"+name+".p", "rb"))

    def loadLibrary(self, name):
        if (name == 'DBUMLS'):
            self.name = 'DBUMLS'
            self.loadDefinitions(UMLSTableIterator("select cui, str from MRCONSO where LAT='ENG' AND SUPPRESS='N'"))
        elif (name == 'FILEUMLS'):
            self.name = 'FILEUMLS'
            self.loadDefinitions(FileIterator(cfg.PATH_RESOURCES+'allUmls.txt',splitTokens=False))
        elif (name == 'DSM'):
            self.name = 'DSM'
            self.loadDefinitions(FileIterator(cfg.PATH_RESOURCES+'dsm4_rel1.txt',splitTokens=False))


    def loadDefinitions(self, firstIterator):
        dontDefine = ['if', 'me', 'any', 'end', 'call', 'po','an', 's', 'on', 'as','sd', 'range','n','i','a','q' 'score', 'all', '=','x','more','may','page','sign','most' '<','16','sum', '>', 'q', 'mi', 'po', 'v', 'td', 's', 'xr', 'h' 'for', 'a', 'axis', 'assessment', 'iv', 'iii', 'd', '31', 'is', 'today', 'at', 'co', 'none', 'one', 'two', 'law', 'yes', 'is', 'at', 'his', 'her', 'who','does']
        dontDefine = set(dontDefine)
        print("Initialising dictionary")
        startT = time.time()
        dfs = dict() #document frequency
        docCount = 0

        #writing to a temporary file.. (because we iterate twice over each definition)
        ft = open(cfg.PATH_TEMP + "tempConcepts.txt", 'w')
        for row in firstIterator: #getting from a table (for now)..
            docCount += 1
            #making a unique identifier from each definition
            cui = row[0] + "-" + str(docCount)
            tokens = set([x.lower() for x in utils.dumbTokeniser(row[1], keepTokens=False)])
            tokenywokeny='exceptionally'
            if len(tokens) == 1:
                for t in tokens:
                    tokenywokeny = t
            
            if ((len(tokens) > 1) | (tokenywokeny not in dontDefine)):
                ft.write(cui + ";" + ",".join(tokens) + "\n")
            ##we add each word, with its definition, to the library
            self.conceptContent[cui] = tokens
            for token in tokens:
                try:
                    relBin = self.tree[token]
                except:
                    relBin = set()
                #if not making a new record in relBin, we do not need to read it to the dictionary, and we cannot increase the count for that token
                relBin.add(cui)
                self.tree[token] = relBin
                #adding it to a normal count for now (since using a set, this is unique per document)
                try:
                    dfs[token] = dfs[token] + 1
                except:
                    dfs[token] = 1
        ft.close()
        print("Got document frequencies and word identifiers")

        ##we calculate the idfs for each word
        for word in dfs:
            self.idfs[word] = math.log(docCount/dfs[word])
            #print(self.idfs[word])
        ##then, we prune the words with an idf score that is below the threshold
        blacklist = []
        for word in self.tree:
            try:
                if (self.idfs[word] < self.retentionThreshold):
                    blacklist.append(word)
            except:
                print("No idf score found for " + word + " while pruning WordDef tree");
        for word in blacklist:
            self.tree.pop(word)
        print("Calculated idf scores and removed below-threshold things")
        
        #writing idf scores to file
        fout = open(cfg.PATH_TEMP+"idf_scores.txt",'w')
        for token in self.idfs:
            fout.write(token.replace(",","") + "," +  str(self.idfs[token])+'\n')
        fout.close()

        ##then, we calculate the complete scores for each concept
        for line in FileIterator(cfg.PATH_TEMP + "tempConcepts.txt"):
            score = 0
            mismatch = 0
            for token in line[1]:
                try:
                    score += self.idfs[token]
                except:
                    mismatch += 1
            self.conceptScores[line[0]] = score
        stopT = time.time()
        print("Initialised dictionary in ",stopT-startT,"seconds!")

    def _calculate_score_table(self, tokens, window_size):

        table = [Counter()] * len(tokens)

        for idx, token in enumerate(tokens[window_size:]):

            try:
                cuis = self.tree[token]
            except KeyError:
                continue

            c = Counter({cui: self.idfs[token] / self.conceptScores[cui] for cui in cuis})

            for idx_2 in range(window_size):
                table[idx - idx_2].update(c)

        return table

    def window(self, tokens, table, windowsize):

        output = []

        for idx, scores in enumerate(table):

            results = {}
            local_tokens = tokens[idx: idx+windowsize]

            for cui, score in scores.items():

                if score > self.threshold:

                    name = cui.split("-")[0]

                    try:
                        prev = results[name]
                        if prev[0] < score:
                            results[name] = [score, len(self.conceptContent[cui]), self.conceptContent[cui]]
                    except KeyError:
                        results[name] = [score, len(self.conceptContent[cui]), self.conceptContent[cui]]

                results2 = []
                for key, value in results.items():
                    # filtering out definitions that only have words matched smaller than 3 letters
                    for word in value[2]:
                        word = word.lower()
                        if word in local_tokens: # and len(word) > 3:
                            results2.append([key.split('-')[0], value])
                            break

                # print(results)
                # results = sorted(results, key=-op.itemgetter(1))
                # results2 has format [[cui,[certainty_score, num of tokens in concept, tokens]]
                # sorts by concept certainty and then by the num of tokens in the concept in descending order
                # results2.sort(key=lambda x: (x[1][0], x[1][1]), reverse=True)
                output.extend((local_tokens, res) for res in results2)

        return output


    def calculateScore(self, tokens):
        ## getting scores for all unique concepts
        noMatch = 0
        localScores = defaultdict(float)
        
        tokens = set(tokens)

        for token, count in Counter(tokens).items():
            try:
                token = token.lower()
                cuis = self.tree[token]
                for cui in cuis:
                    localScores[cui] += self.idfs[token] * count
            except KeyError:
                noMatch += 1
        results = dict()
        for identifier, score in localScores.items():
            try:
                scoren = score / self.conceptScores[identifier]
                name = identifier.split('-')[0]
                if scoren > self.threshold:
                    try:
                        prev = results[name]
                        if prev[0] < scoren:
                            results[name] = [scoren, len(self.conceptContent[identifier]), self.conceptContent[identifier]]
                    except KeyError:
                        results[name] = [scoren, len(self.conceptContent[identifier]), self.conceptContent[identifier]]
            except KeyError:
                continue

        #these were post-processing steps to filter out definitions, not used atm
        # get raw definitions
        #tokens = set(tokens)

        results2 = []
        for key, value in results.items():
            #bypass
            results2.append([key.split('-')[0], value])
            
            # filtering out definitions that only have words matched smaller than 3 letters (disabled due to interest in abbreviations!)
            '''for word in value[2]:
                word = word.lower()
                if word in tokens and len(word) > 1:
                    results2.append([key.split('-')[0], value])
                    break'''

        # print(results)
        # results = sorted(results, key=-op.itemgetter(1))
        # results2 has format [[cui,[certainty_score, num of tokens in concept, tokens]]
        # sorts by concept certainty and then by the num of tokens in the concept in descending order
        results2.sort(key=lambda x: (x[1][0], x[1][1]), reverse=True)
        
        return results2

    def process_text_cTakes(self, text, use_chunks=True, tokenized = True, use_begin_end=False):
        # check each chunk for concepts present
        results = []
        if use_chunks:
            for chunk in text.get_chunks(pos=False, with_begin_end=use_begin_end):
                res = self.calculateScore(chunk[0])
                if len(res):
                    for concept in res:
                        results.append((chunk, concept))
        else:
            if tokenized:
                tokens = text.get_tokens()
            else:
                tokens = text.split(' ')

            tokens = [x for x in tokens if x not in {'',' ', '\s','<newline>',',','.','<tab>',':','\n',';','?','!','-','\\','"',"'",')','_','*'}]

            table = self._calculate_score_table(tokens, window_size=5)
            results = self.window(tokens, table, windowsize=5)

            '''for window in WindowSlider(tokens, 5):

                res = self.calculateScore(window)
                if len(res):
                    for concept in res:
                        results.append(((window, concept)))'''

        return results

    def process_text(self, text, use_chunks=True, tokenized = True, use_begin_end=False):
        # check each chunk for concepts present
        results = []
        if use_chunks:
            for chunk in text.get_chunks(pos=False, with_begin_end=use_begin_end):
                res = self.calculateScore(chunk[0])
                if len(res):
                    for concept in res:
                        results.append((chunk, concept))
        else:
            if tokenized:
                tokens = text.get_tokens()
            else:
                tokens = text.split(' ')

            tokens = [x for x in tokens if x not in {'',' ', '\s','<newline>',',','.','<tab>',':','\n',';','?','!','-','\\','"',"'",')','_','*'}]

            #table = self._calculate_score_table(tokens)
            #results = self.window(tokens, table, windowsize=5)

            for window in WindowSlider(tokens, 5):

                res = self.calculateScore(window)
                if len(res):
                    for concept in res:
                        results.append(((window, concept)))

        return results

    def getListOfConcepts(self, text, basedOnChunks = True, tokenized = False):
        det = self.process_text_cTakes(text, basedOnChunks, tokenized=tokenized)
        res = []
        for d in det:
            res.append(d[1][0])
        return res

    def getXMLListOfConcepts(self, text, basedOnChunks = True):
        res = []
        dets = self.process_text_cTakes(text, basedOnChunks, use_begin_end=True)
        for det in dets:
            #find all overlapping words
            commonPool = set()
            for word in det[0][0]:
                word = word.lower()
                if word in det[1][1][2]:
                    commonPool.add(word)
            #print(det)
            res.append("<concepts_FILEUMLS begin='"+str(det[0][2])+"' end='"+str(det[0][3])+"' identifier='"+str(det[1][0])+"' certainty='"+str(det[1][1][0])+"' words='"+','.join(commonPool)+"' />")
        return res
    
def remove_present_concepts(text):
    
    #<concepts_FILEUMLS begin='10103' end='10117' identifier='C1691010' certainty='1.0' words='referred' />
    
    m = re.findall ( '<concepts_FILEUMLS(.*?)\/>', text, re.DOTALL)
    for n in m:
        text = text.replace("<concepts_FILEUMLS"+n+"/>\n","")
        text = text.replace("<concepts_FILEUMLS"+n+"/>","")
    return text

if __name__ == '__main__':
    #cfg.makeFolders()
    #texts = utils.readData(cfg.PATH_INPUT)
    #for text in texts:
    #    texts[text]["tokens"] = utils.dumbTokeniser(texts[text]["note"])
    
    basic = cfg.PATH_TEST
    preprocessed = cfg.PATH_PREPROCESSED_TEST
    
    data = utils.readData(basic, preprocessed)
    
    #TODO reset this to 0.80 idf threshold
    matcher = DictionaryMatcher(4, 0.80)
    matcher.loadLibrary('FILEUMLS')
    #matcher.loadLibrary('DSM')
    # matcher.loadDefinitions()
    
    #matcher.saveModel()
    #matcher = matcher.loadModel('FILEUMLS')
    
    for d in data:
        text = d.getTextObject()
        #for line in matcher.processText(text, True):
        #    print(line)
        
        content = ''
        ##get original file
        print(d.key)
        fin = open(preprocessed + d.key + '.xml', 'r');
        # TODO
        ##replace </xmi:XMI>
        for line in fin:
            content += line

        content = remove_present_concepts(content)
        content = content.replace('</xmi:XMI>','')
        content = content.replace('><','>\n<')
        
        ##add concepts
        for concept in matcher.getXMLListOfConcepts(text, True):
            content += concept 
            content += '\n'
        content += '</xmi:XMI>'
        
        fout = open(preprocessed + d.key + '_concepts.xml', 'w');
        fout.write(content)
        fout.close()
