'''
Created on 5 July 2016

@author: madhumita
'''

import config as cfg
import utils

def createMedList(fname):
    
    utils.out("Opening file to store medication list and their relations with active compound")
    out_file = open(fname,'w')
    
    utils.out("Getting UMLS connection")
    umls_conn = cfg.connUMLS()
    cur1 = umls_conn.cursor()
    cur2 = umls_conn.cursor()
    cur3 = umls_conn.cursor()
    
    utils.out("Executing query")
    cur1.execute("SELECT CUI,STR FROM umls.MRCONSO where sab='RXNORM'")
    for i, cur_med in enumerate(cur1.fetchall()):
        cur2.execute("SELECT DISTINCT CUI2 FROM umls.MRREL where cui1=%s",(cur_med[0],))
        for cui2 in cur2.fetchall():
            '''
            Finding ATC links for all relations to CUI1 in MRREL because direct link from CUI1 to ATC component not always present in MRREL.
            However, finding a corresponding ATC entry from MRCONSO solves the problem.
            '''
            cur3.execute("SELECT DISTINCT STR FROM umls.MRCONSO where cui=%s and sab ='ATC'",(cui2[0],))
            for j,row in enumerate(cur3.fetchall()):
                med = cur_med[0]+'-'+str(i)+';'+cur_med[1]+';'+cui2[0]+'-'+str(j)+';'+row[0]+'\n'
                out_file.write(med)
                
    out_file.close()
        
def main():
    createMedList('../'+cfg.PATH_RESOURCES+'all_rxnorm.txt')
    
if __name__ == "__main__":
    main()
        
    
    
    
    
    

