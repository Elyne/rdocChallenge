#import sys
#sys.path.insert(0,"../")

import re
import config as cfg
import utils
import os

def getUpdatedStr(regex, regex_exclude, found, splitType, string):
    new_str = ""
    first = True

    for cur_re in regex:
        end = 0
        for match in cur_re.findall(string):
            found = True
            t_idx = string.index(match)
                
            if splitType == 1 or splitType ==3:
                if splitType ==3:
                    if regex_exclude.search(string) != None:
                        found = False
                        continue
                if first and t_idx != 0:
                    new_str += string[0:t_idx]+"\n"
                first = False
                new_str += string[t_idx:len(match)+t_idx]+"\n"
                end = len(match)+t_idx
                        
            elif splitType == 2:
                    
                if t_idx!=0:
                    new_str += string[0:t_idx]+"\n"
                        
                new_str += string[t_idx : t_idx + len(match)-1]
                new_str += "\n"
                new_str += string[t_idx + len(match)-1 : t_idx + len(string)]+"\n"
                
        if end and end != len(string):
            new_str += string[end:len(string)]
    return (new_str.strip(), found)

regex_exclude = re.compile('[\s]*[-(]?[A-Z0-9][a-z0-9]+[-/][A-Z0-9][a-z0-9]+[):]*[\s]*')
regex1 = re.compile('[-(]?[A-Z][a-z0-9]+[-/]?[A-Z]?[^A-Za-z0-9]?[a-z0-9):]*[\s]*')
# regex1 = re.compile('[-(]?[A-Z][a-z0-9]+[^A-Za-z0-9]?[a-z0-9):]*[\s]*')
regex2 = re.compile("[/W0-9^0-9.0-9]*[a-z0-9]+[A-Z]")
# regex2 = re.compile("[^A-Z\"]+[A-Z][^\'/,:)][\W^\'/)]?[^A-Z][a-z0-9]*[\s]*")
regex3 = re.compile("HPI[^$\W]")
regex4 = re.compile("SYRINGE[^$\W]")
# regex4 = re.compile("[\s]+DEPRESSION")
regex5 = re.compile("PTSD[^$\W]")

regex_last = re.compile('[-(]?[A-Z][a-z0-9]+[\W]?[a-z0-9:]*[\s]*')

# data = utils.readData("../"+cfg.PATH_INPUT, "../"+cfg.PATH_PREPROCESSED_TRAIN, 1)

outDir = cfg.PATH_TRAIN+"refactor/"

try:
    os.makedirs(outDir)
except OSError:
    pass

data = utils.read_content(cfg.PATH_TRAIN)

for idx, content in data.items():
    content_str = ""
    for word in content.split(' '):
        found = False
        
        new_str, found = getUpdatedStr([regex1], regex_exclude, found, 1, word)
#         if not found:
        new_str,found = getUpdatedStr([regex_last], regex_exclude,found, 3, word)
        if not found:
            new_str, found = getUpdatedStr([regex2, regex3, regex4, regex5], regex_exclude, found, 2, word)
        if not found:
            new_str = word
        content_str += " "+new_str
    
#         print("final content: ",content_str) 
    
    fout = open(outDir+idx[0:idx.index(".xml")]+".txt","w")
    fout.write(content_str)
    fout.close()