'''
Created on 15 Jun 2016

'''
import config as cfg
import utils
#import modelData as m
#from preprocessing.Segment import Segment
#from QandAFeatures import QandAFeatures

umls = cfg.connUMLS()

def printDefinition(cui):
    cursor = umls.cursor()
    cursor.execute("select STR, SAB from MRCONSO where CUI='"+cui+"'")
    row = cursor.fetchone()
    d = []
    while row:
        d.append(', '.join(row))
        row = cursor.fetchone()
    utils.out('; '.join(d))
    
def fetchDefinition(cui):
    cursor = umls.cursor()
    cursor.execute("select STR, SAB from MRCONSO where CUI='"+cui+"'")
    row = cursor.fetchone()
    while row:
        return row[0]
        row = cursor.fetchone()
        
def getRelations(cui, rela=None):
    cursor = umls.cursor()
    if rela is None:
        cursor.execute("select REL, RELA, CUI2, SAB from MRREL where CUI1='"+cui+"'")
    else:
        cursor.execute("select REL, RELA, CUI2, SAB from MRREL where CUI1='"+cui+"' AND RELA='"+rela+"'")
    res = []
    row = cursor.fetchone()
    while row:
        res.append(row)
        row = cursor.fetchone()
    return res


def getDescendants(descendants, cui, depth, maxDepth=10):
    if (depth < maxDepth):
        rels = getRelations(cui,"isa")
        depth += 1
        for rel in rels:
            if (rel[2] not in descendants):
                #print(rel[2],"is already present in descendants! Not checking it further")
                #else:
                if (rel[3] == 'SNOMEDCT_US'):
                    descendants[rel[2]] = depth
                    descendants.update(getDescendants(descendants, rel[2],depth, maxDepth))
    return descendants


if __name__ == '__main__':
    umls = cfg.connUMLS()
    
    root = 'C1291705'
    
    utils.out("ROOT")
    printDefinition(root)
    
    descendants = dict()
    children = getDescendants(descendants, root,0, 12)
    print(len(children), "descendants found with a maximum of ",12," steps!")
        
    fout = open(cfg.PATH_RESOURCES + 'SNOMED_PSYCH.txt', 'w')
    for anc in descendants:
        fout.write(anc + "-" + str(descendants[anc]) + ";" + fetchDefinition(anc) + '\n')
    fout.close()
    
    
    nd = dict()
    for anc in descendants:
        nd[anc] = descendants[anc]
        for dev in getRelations(anc):
            if (dev[2] not in nd):
                nd[dev[2]] = 100        
    
    fout = open(cfg.PATH_RESOURCES + 'SNOMED_PSYCH+1.txt', 'w')
    for anc in nd:
        fout.write(anc + "-" + str(nd[anc]) + ";" + fetchDefinition(anc) + '\n')
    fout.close()
    
    
    
    utils.out("CHILDREN")
    rels = getRelations(root,"isa")
    for rel in rels:
        descendants = dict()
        children = getDescendants(descendants, rel[2],0, 12)
        print(len(children), "descendants found with a maximum of ",12," steps!")
        
        if (len(children) > 50):
            fout = open(cfg.PATH_RESOURCES + 'SNOMED_PSYCH_'+fetchDefinition(rel[2]).replace('\\',' ')+'.txt', 'w')
            for anc in descendants:
                fout.write(anc + "-" + str(descendants[anc]) + ";" + fetchDefinition(anc) + '\n')
            fout.close()
    
        