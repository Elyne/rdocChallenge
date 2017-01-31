'''
This file is part of Treecounter.

Treecounter is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Treecounter is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Treecounter.  If not, see <http://www.gnu.org/licenses/>.

This file is a collection of functions that allows us to make predictions on data.


Created on 3 Jun 2016

'''

import os
import re
import xml.etree.ElementTree as ET
import Annotations
from Text import Text
from Data import Data
from Annotations import Token
from io import open
import datetime
import sys

splitter = re.compile(r'(\s|<newline>|\.|<tab>|:|,|;|\?|!|-|\"|\'|/|\))')

def dumbTokeniser(text, keepTokens=True):
    tokens = splitter.split(text)
    if not keepTokens:
        nTokens = []
        for token in tokens:
            if token not in {'',' ', '\s','<newline>',',','.','<tab>',':','\n',';','?','!','-','\\','"',"'",')','_','*','/'}:
                nTokens.append(token)
        return nTokens
    return tokens


def match2span(token2idx, begin, end):

    b_indices, e_indices, tokens = token2idx

    try:

        b = b_indices[begin]
        e = e_indices[end]+1
        return tokens[b:e]

    except KeyError:

        output = []
        inchunk = False

        for w_index, w in b_indices.items():
            if begin <= w_index < end:
                output.append(tokens[w])
                inchunk = True
            else:
                if inchunk:
                    break

        return output


def get_sentences(ns, root, token2idx):
    
    sentences = []
    
    sentAnnots = root.findall('textspan:Sentence', ns)
    
    for sentence in sentAnnots:

        begin = int(sentence.get('begin'))
        end = int(sentence.get('end'))

        tokens = match2span(token2idx, begin, end)

        sentences.append(Annotations.Sentence(begin, end, tokens))
    
    return sentences


def get_tokens(ns, root, annotation_type, text):
    
    if annotation_type == "word":
        annots = root.findall('syntax:WordToken', ns)
    elif annotation_type == "punctuation":
        annots = root.findall('syntax:PunctuationToken', ns)
    elif annotation_type == "symbol":
        annots = root.findall('syntax:SymbolToken', ns)
    elif annotation_type == "number":
        annots = root.findall('syntax:NumToken', ns)
    elif annotation_type == "contraction":
        annots = root.findall('syntax:ContractionToken', ns)
    else:
        raise ValueError("Invalid annotation type.")
    
    tokens = []
    for cur_annot in annots:

        begin = int(cur_annot.get('begin'))
        end = int(cur_annot.get('end'))
        pos = cur_annot.get('partOfSpeech')
        string = text[begin:end]
        tokens.append(Annotations.Token(begin, end, string, pos))
    
    return tokens


def get_chunks(ns, root, token2idx):
    
    chunks = []
    
    chunk_annotations = root.findall('syntax:Chunk', ns)
    
    for curChunk in chunk_annotations:

        begin = int(curChunk.get('begin'))
        end = int(curChunk.get('end'))
        chunk_type = curChunk.get('chunkType')

        tokens = match2span(token2idx, begin, end)
        # TODO: link chunk to token

        chunks.append(Annotations.Chunk(begin, end, tokens, chunk_type))
    
    return chunks


def get_concepts(root, token2idx):
    concepts = []
    
    conceptAnnots = root.findall('concepts_FILEUMLS')

    for cur_annot in conceptAnnots:

        begin = int(cur_annot.get('begin'))
        end = int(cur_annot.get('end'))
        ide = cur_annot.get('identifier')
        certainty = float(cur_annot.get('certainty'))
        tokens = match2span(token2idx, begin, end)

        cur_concept = Annotations.Concept(begin, end, ide, certainty, tokens)
        
        concepts.append(cur_concept)
    
    return concepts


def get_annotations(fName):
    ns = {'refsem': 'http:///org/apache/ctakes/typesystem/type/refsem.ecore',
      'cas': 'http:///uima/cas.ecore', 
      'textspan': 'http:///org/apache/ctakes/typesystem/type/textspan.ecore', 
      'syntax': 'http:///org/apache/ctakes/typesystem/type/syntax.ecore',
      'textsem': 'http:///org/apache/ctakes/typesystem/type/textsem.ecore'}
    
    f = open(fName, encoding='UTF-8')

    root = ET.parse(f).getroot()

    for cas in root.findall('cas:Sofa', ns):
        for attr in cas.attrib:
            if attr == 'sofaString':
                content = cas.get(attr)

    tokens = get_tokens(ns, root, "word", content)
    tokens.extend(get_tokens(ns, root, "punctuation", content))
    tokens.extend(get_tokens(ns, root, "symbol", content))
    tokens.extend(get_tokens(ns, root, "number", content))
    tokens = sorted(tokens, key=lambda x: x.begin)

    begins, ends, tokens = zip(*[(x.begin, x.end, x) for x in tokens])
    b_indices = {k: idx for idx, k in enumerate(begins)}
    e_indices = {k: idx for idx, k in enumerate(ends)}

    token2idx = (b_indices, e_indices, tokens)

    chunks = get_chunks(ns, root, token2idx)
    concepts = get_concepts(root, token2idx)
    sentences = get_sentences(ns, root, token2idx)

    return Text(content, chunks, tokens, sentences, concepts)


def get_dumb_tokens(text):

    cnt = 0

    tokens = []
    
    for token in dumbTokeniser(text):

        str2 = token
        begin = cnt
        end = cnt + 1
        cnt += 1
        tokens.append(Token(begin, end, str2, ""))

    return Text(tokens=tokens, concepts=[], sentences=[], chunks=[], content=text)


def readXML(fname):

    if(fname.endswith(".xml")):
        f = open(fname)
        
        severity = None
        Nannotators = str(0)
    
        root = ET.parse(f).getroot() 
        for child in root:
            if child.tag == 'TEXT':
                tmpText = child.text
            if child.tag == 'TAGS': #severity grade
                for toddler in child:
                    if toddler.tag == 'POSITIVE_VALENCE':
                        try:
                            severity = toddler.attrib['score']
                            if toddler.attrib['score'] == 'ABSENT':
                                severity = 0
                            elif toddler.attrib['score'] == 'MILD':
                                severity = 1
                            elif toddler.attrib['score'] == 'MODERATE':
                                severity = 2
                            elif toddler.attrib['score'] == 'SEVERE':
                                severity = 3
                            else:
                                print("Label", toddler.attrib['score'], "not recognised!")
                                
                            Nannotators = toddler.attrib['annotated_by']
                        except KeyError:
                            print('No tags were found for text', fname)

    return severity, Nannotators, tmpText
    
    
def readData(inputData, preProcessedData, limit=10000):
    
    data = []
    i = 0
    
    fNames = os.listdir(inputData)
    nFiles = min(len(fNames), limit)
    
    for fname in fNames:
        if fname.endswith('.xml'):

            severity, n_annotators, content = readXML(os.path.join(inputData, fname))
            try:
                text = get_annotations(os.path.join(preProcessedData, fname))
            except FileNotFoundError:
                print('Text {0} was not pre-processed with the cTakes pipeline; please do so.'.format(fname))
#                 text = get_dumb_tokens(content)
                continue
            except UnicodeDecodeError:
                print("Error reading file")
                text = get_dumb_tokens(content)
            
            text.content = content

            key = os.path.splitext(fname)[0]

            data.append(Data(text, severity, n_annotators, key))
            
            if (nFiles >= 10):
                if (i % int(nFiles/10) == 0):
                    printProgress(i, nFiles , prefix = 'Loading data:', suffix = 'Complete', barLength = 50)
            i += 1
            if i == limit:
                break
        
    return data


def read_content(input_data, limit=1000):
    data = {}
    
    for idx in os.listdir(input_data):
        if idx.endswith(".xml"): 
            f = open(input_data + idx)
    
            root = ET.parse(f).getroot() 
            for child in root:
                if child.tag == 'TEXT':
                    data[idx] = child.text      
        
    return data
    

def out(message):
    print(datetime.datetime.now().strftime("%H:%M:%S"),'| ', message)


def writeXML(dataObj, fname, dtdPath):
    
    if dataObj.predSev == 0:
        sev = "ABSENT"
    elif dataObj.predSev == 1:
        sev = "MILD"
    elif dataObj.predSev == 2:
        sev = "MODERATE"
    elif dataObj.predSev == 3:
        sev = "SEVERE"
            
    root = ET.Element("RDoC")
        
    ET.SubElement(root, "TEXT").text = dataObj.text.content
    tags = ET.SubElement(root, "TAGS")
        
    pos_val = ET.SubElement(tags, "POSITIVE_VALENCE")
    pos_val.set('score', sev)
    pos_val.set('annotated_by', dataObj.Nannotators)
    
    tree = ET.ElementTree(root)

    tree.write(fname)


def genOutput(data,outDir,dtd):
    for d in data:
        ide = d.key
#         txtObj = d.text
        writeXML(d,outDir+ide+'.xml',dtd)
        
def find_between( s, start, end ):
    return re.findall(re.escape(start)+"(.*)"+re.escape(end),s)[0]
        
# Print iterations progress
def printProgress (iteration, total, prefix = '', suffix = '', decimals = 2, barLength = 100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    filledLength    = int(round(barLength * iteration / float(total)))
    percents        = round(100.00 * (iteration / float(total)), decimals)
    bar             = '█' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    sys.stdout.flush() 
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()
