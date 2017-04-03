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

Created on 10 Jun 2016

'''
import config as cfg
import numpy as np
from scipy import spatial
from ExperimentSettings import ExperimentSettings
from Resources import Resources
import math

class UMLSFeatures(object):
    '''
    UMLS features is a collection of feature functions all involving subsets of UMLS in some way.
    Hierarchic distance is also defined here, as this is calculated through UMLS
    
    '''
    
    def __init__(self, expSet=ExperimentSettings(), path_to_vector=""):
        '''
        Constructor
        '''
        try:
            self.conn = cfg.connUMLS()
        except:
            self.conn = None
            
        self.expSet = expSet
        self.hierCache = dict()
        
    def cuiFilter(self, cui, includePreambled):
        if (includePreambled):
            return cui.replace("DEN_","").replace("UNC_","").replace("FAM_","")
        else:
            return cui
        
    def isPsychiatric(self, cui, lexiconName, includePreambled=True):
        '''
        From the prespecified lexicon, find if a cui can be considered psychiatric
        '''   
        pureCUI = self.cuiFilter(cui, includePreambled)
        if (pureCUI in Resources.getLexiconFilter(lexiconName)):
            return (True, cui+";"+Resources.getLexiconFilter(lexiconName)[pureCUI])
        return (False, None)

    '''def isPsychiatric(self, cui, includePreambled=True):
        
        pureCUI = self.cuiFilter(cui, includePreambled)
        if (pureCUI in Resources.getDSM()):
            return (True, cui+";"+Resources.getDSM()[pureCUI])
        return (False, None)
    
    def isSnomedDefined(self, cui, includePreambled=True):
        ''
        From the internal list, find if a cui can be considered psychiatric
        ''   
        pureCUI = self.cuiFilter(cui, includePreambled)
        if (pureCUI in Resources.getSnomedFilter()):
            return (True, cui+";"+Resources.getSnomedFilter()[pureCUI])
        return (False, None)
    
    def isRemotelyPsychiatric(self, cui, includePreambled=True):
        ''
        From the internal list, find if a cui can be considered psychiatric, or one step away from it 
        (because of it being a concept related to psychiatric)
        ''
        pureCUI = self.cuiFilter(cui, includePreambled)
        if pureCUI  in Resources.getDSMdev1():
            return (True, cui+";"+Resources.getDSMdev1()[pureCUI])
        return (False, None)'''
    
    def calculateAverageHierarchicDistance(self, cui, others):
        '''
        This function uses recursive steps to determine the closest common ancestor of cuis and calculates the average distance between all of them.
        '''
        
        def distanceBetween(cui1, cui2):  
            if (cui1 == cui2):
                return 0
            elif cui1+'-'+cui2 in self.hierCache:
                #check cache
                return self.hierCache[cui1+'-'+cui2]
            elif cui2+'-'+cui1 in self.hierCache:
                return self.hierCache[cui2+'-'+cui1]
            else:
                bin1 = dict()
                bin1[cui1] = 0
                bin2 = dict()
                bin2[cui2] = 0
                return recursiveDistanceBetween(bin1, bin2, 0)
            
        def recursiveDistanceBetween(bin1=dict(), bin2=dict(), steps=0):
            #find common ancestor, by first getting all ancestors of both nodes, and their distance from the node (first go up one level, then expand)
            if (steps > 5):
                print("More than five steps, don't count on it")
                print(bin1)
                print(bin2)
                return None
    
            ancs = dict()
            for cui in bin1:
                dist = bin1[cui]
                if (dist == steps):
                    for anc in getAncestors(cui):
                        ancs[anc] = dist + 1
            for anc in ancs:
                bin1[anc] = ancs[anc]
                
            ancs = dict()
            for cui in bin2:
                dist = bin2[cui]
                if (dist == steps):
                    for anc in getAncestors(cui):
                        ancs[anc] = dist + 1
            for anc in ancs:
                bin2[anc] = ancs[anc]
                   
            for rec in bin1:
                if rec in bin2:
                    #calculate sum
                    return bin1[rec] + bin2[rec] + .0
            steps+=1
            return recursiveDistanceBetween(bin1, bin2, steps)
        
        def getAncestors(cui):
            anc = set()
            cursor = self.conn.cursor()
            cursor.execute("select cui2 from MRREL where CUI1='" + str(cui) + "' AND REL='PAR'")
            row = cursor.fetchone()
            while row:
                anc.add(row[0])
                row = cursor.fetchone()
            return anc
        
        avg = 0
        cnt = 0
        maxx = 0
        if len(others) > 0:
            for cui2 in others:
                val = distanceBetween(cui, cui2)
                self.hierCache[cui+'-'+cui2] = val
                #print("distance:",val)
                if (val != None):
                    avg += val
                    cnt += 1
                    if (maxx < val):
                        maxx = val
            if (cnt > 0):
                return [avg / cnt, maxx]
        return [0,0]
    
    def kill(self):
        self.conn.close()
    
    
    def getUMLSFeatures(self,concepts):
        '''
        Converts the predetected concepts we have to a list of features, that is not filtered in any way
        ''' 
        feats = dict()
        for concept in concepts:
            try:
                val = feats[concept] + 1
            except:
                val = 1
            feats[concept] = val
        return feats
    
    
    '''def getDSMPlusOneFeatures(self, concepts, getConcepts=True, getMeta=True, verbose=False):
        ''
        Calculates DSM+1-based features
        
        Does NOT calculate hierarchical distance, as this requires concepts within a given framework
        ''
        feats = dict()
        for cui in concepts:
            #by adding .replace("DEN_",""), we allow denied concepts (if any), to also be detected
            is_psych, cur_concept = self.isRemotelyPsychiatric(cui)
            
            if (is_psych):
                try:
                    val = feats[cur_concept] + 1
                except:
                    val = 1
                if getConcepts:
                    feats[cur_concept] = val
        
        ##if meta, add the number of concepts
        if getMeta:
            feats['numDSM+1Concepts'] = len(feats)
        if (verbose):
            print(len(feats),'of',len(concepts), 'are defined in the DSM+1!')
        return feats'''
    
    
    def getLexiconFeatures(self, concepts, lexiconName, getConcepts=True, getMeta=True, verbose=False):
        '''
        Calculates all psych-based features
        
        Does not calculate the hierarchical distance metric anymore
        That metric should be used on features from a limited type (e.g. only diagnostics, only medication, ..), otherwise
        it is not representative of the distance between facts, only of how the umls is defined.    
        '''
        pfeats = dict()
        
        ##get the concepts defined in DSM (only contains diagnostic concepts)
        for cui in concepts:
            is_psych, cur_concept = self.isPsychiatric(cui, lexiconName)
            
            if (is_psych):
                if getConcepts:
                    try:
                        val = pfeats[cur_concept] + 1
                    except:
                        val = 1
                    pfeats[cur_concept] = val
        
        ##if meta, add the number of concepts
        if getMeta:
            pfeats['numPsychConcepts'] = len(pfeats)
        if (verbose):
            print(len(pfeats)-1,'of',len(concepts), 'are defined in the DSM!')
            
        return pfeats
    
    
    def getSubsetOfConcepts(self, concepts, filterName='DSM'):
        subset = []
        if (filterName == 'DSM+1'): #only perform for concepts within dsm+1 range
            for cui in concepts:
                is_psych, cName = self.isPsychiatric(cui, filterName, includePreambled=False)
                if (is_psych):
                    subset.append(cName)
        elif (filterName == 'MED'):
            for cui in concepts:
                is_active, cur_concept = self.isActiveComponent(cui, includePreambled=False)
                if is_active:
                    subset.append(cur_concept)
                else:
                    if self.isMedBrand(cui):
                        active_comp_list = self.getActiveComponent(cui, includePreambled=False)
                        for active_comp in active_comp_list:
                            subset.append(active_comp)
        else: 
            #only perform for concepts within dsm range (DEFAULT filterName=DSM)
            for cui in concepts:
                is_psych, cName = self.isPsychiatric(cui, includePreambled=False)
                if (is_psych):
                    subset.append(cName)
        return subset
    
    def getHierarchicalDistanceFeatures(self, concepts, filterName='DSM'):
        '''
        we calculate the average distance of all concepts from each other (based on the hierarchical distance- only going up/down the hierarchy)
        '''
        hierfeats = dict()
        if (filterName=='ALL'):
            hierfeats.update(self.getHierarchicalDistanceFeatures(concepts, 'DSM'))
            hierfeats.update(self.getHierarchicalDistanceFeatures(concepts, 'MED'))
        else:
            concepts = self.getSubsetOfConcepts(concepts, filterName) 
            psycs = set()
            for concept in concepts:               
                psycs.add(concept.split(';')[0])           
            
            avg = 0
            avgMax = 0
            maxx = 0
            psycsdist = []
            if (len(psycs) > 0):
                for cui in psycs:
                    val = self.calculateAverageHierarchicDistance(cui, psycs)
                    psycsdist.append(val)
                    #print("avgForPoint:",val)
                    avg += val[0]
                    avgMax += val[1]
                    if (maxx < val[1]):
                        maxx = val[1]
                avg = avg/len(psycs)
                avgMax = avgMax/len(psycs)
                
                hierfeats[filterName + 'hierAvgDist'] = avg
                hierfeats[filterName + 'hierAvgMaxDist'] = avgMax
                hierfeats[filterName + 'hierMaxDist'] = maxx
        return hierfeats
    
    def calculateAverageWordVectorDistance(self, concepts, filterName='DSM'):
        '''
        This function calculates the average word vector distance, based on the words that a concept is encompassed by.
        
        '''
        feats = dict()
        if (filterName=='ALL'):
            feats.update(self.calculateAverageWordVectorDistance(concepts, 'DSM'))
            feats.update(self.calculateAverageWordVectorDistance(concepts, 'DSM+1'))
            feats.update(self.calculateAverageWordVectorDistance(concepts, 'MED'))
        else:
            subset = list(set(self.getSubsetOfConcepts(concepts, filterName)))
            maxDist = 0
            dists = []
            
            for cui in subset:
                cumulDist = 0
                cnt = 0
                for cui2 in subset:
                    try:
                        words = cui.split(';')[1].lower().split(' ')
                        wVector= np.mean(Resources.getWordVectors().vectorize(words, remove_oov=True), axis=0)
                        words = cui2.split(';')[1].lower().split(' ')
                        wVector2= np.mean(Resources.getWordVectors().vectorize(words, remove_oov=True), axis=0)
                        cosDist = spatial.distance.cosine(wVector, wVector2)
                        if not math.isnan(cosDist):
                            cumulDist += cosDist
                            cnt += 1
                    except:
                        continue
                try:
                    if (cnt != 0):
                        dists.append(cumulDist/cnt)
                    if (cumulDist/cnt) > maxDist:
                        maxDist = cumulDist/cnt
                except:
                    continue
            
            feats[filterName + 'maxWordVectorDist'] = round(maxDist,3)
            if bool(dists):
                feats[filterName + 'avgWordVectorDist'] = round(np.mean(dists),3)
        return feats
    
    def calculateAverageConceptVectorDistance(self, concepts, filterName='DSM'):
        '''
        This function calculates the average concept vector distance, based on the vector representing a cui.
        
        '''
        feats = dict()
        if (filterName=='ALL'):
            feats.update(self.calculateAverageConceptVectorDistance(concepts, 'DSM'))
            feats.update(self.calculateAverageConceptVectorDistance(concepts, 'DSM+1'))
            feats.update(self.calculateAverageConceptVectorDistance(concepts, 'MED'))
        else:
            subset = list(set(self.getSubsetOfConcepts(concepts, filterName)))
            maxDist = 0
            dists = []
            
            for cui in subset:
                for cui2 in subset:
                    try:
                        concept = cui.split(';')[0]
                        wVector= Resources.getConceptVectors_dsm().vectorize(concept, remove_oov=True)
                        concept2 = cui2.split(';')[0]
                        wVector2= Resources.getConceptVectors_dsm().vectorize(concept2, remove_oov=True)
                        cosDist = spatial.distance.cosine(wVector, wVector2)
                        if not math.isnan(cosDist):
                            dists.append(cosDist)
                        if (cosDist) > maxDist:
                            maxDist = cosDist
                    except:
                        continue
                    
            feats[filterName+'maxConceptVectorDist'] = round(maxDist,3)
            if bool(dists):
                feats[filterName+'avgConceptVectorDist'] = round(np.mean(dists),3)
        return feats
        
    
    def getMedFeats(self, concepts, getMeds = True, getActComp = False, getMeta = True, verbose = False):
        '''
        Calculate bag of medication features: {active-comp:freq}
        Check if a concept present in the data is an active component of a medication. If yes, add it as a feature.
        Else, check if the concept is present as a medication in the RxNorm ontology. If yes, add as features
        # If yes, get corresponding related active compounds from the ATC ontology linked in UMLS and add them as features.
        @param getMeds: whether we want the medication features (can be off if we want only meta features)
        @param getActComp: True to return active compounds for brand names of medicines.
        '''
        
        meds = {}
        for concept in concepts:
            is_active, cur_concept = self.isActiveComponent(concept)
            if is_active:
                meds[cur_concept] = meds.get(cur_concept,0)+1
            else:
                isBrand, brand_concept = self.isMedBrand(concept)
                if isBrand:
                    if getActComp:
                        print("Getting active comp list")
                        active_comp_list = self.getActiveComponent(concept)
                        if len(active_comp_list) == 0: #if no corresponding active component found, use the brand name as feat
                            meds[brand_concept] = meds.get(brand_concept, 0) + 1 
                        for active_comp in active_comp_list:
                            meds[active_comp] = meds.get(active_comp,0)+1
                        
                    else:
                        print("Adding brand name")
                        meds[brand_concept] = meds.get(brand_concept, 0) + 1
               
        if (verbose):
            print(len(meds),'of',len(concepts), 'concepts are unique Medication')     
                           
        #if meta, add the number of concepts
        if getMeta:
            if getMeds:
                meds['numMedConcepts'] = len(meds)
            else:
                return{'numMedConcepts':len(meds)}
        
        return meds
    
    
    def isActiveComponent(self, cui, includePreambled=True):
        '''
        Check if a concept is an active component of a medicine, according to the ATC ontology linked to UMLSs
        '''
        _, activeCompFilter, _ = Resources.getMedFilter()
        pureCUI = self.cuiFilter(cui, includePreambled)
        if pureCUI in activeCompFilter:
            return (True, cui+";"+ activeCompFilter[pureCUI])
        return (False,None)
    
    def isMedBrand(self, cui, includePreambled=True):
        '''
        Check if a concept is a medication as present in RxNorm linked to UMLS.
        '''
        medFilter, _, _ = Resources.getMedFilter()
        pureCUI = self.cuiFilter(cui, includePreambled)
        if pureCUI in medFilter:
            return (True, cui+";"+ medFilter[pureCUI])
        return (False,None)
    
    def getActiveComponent(self, cui, includePreambled=True):
        '''
        Get a list of active components (From ATC ontology in UMLS) related to an RxNorm medication in the UMLS
        '''
        _, activeCompFilter, medRelFilter = Resources.getMedFilter()
        pureCUI = self.cuiFilter(cui, includePreambled)
        if pureCUI not in medRelFilter:
            return []
        cui2_list = medRelFilter.get(pureCUI,[])
        cui2_list = [cui2+";"+activeCompFilter[cui2] for cui2 in cui2_list]
        return cui2_list

    def getVectorizedConcepts(self, concepts):
        """
        Vectorizes a list of Concept Unique Identifiers into an array of concept vectors of identical size.

        @param concepts: A list of CUIS.
        @return: A list of vectors, each one of which represents a concept.
        """

        return Resources.getConceptVectors_dsm().vectorize(concepts, remove_oov=False)

    def getVectorizedWords(self, words):

        return {str(idx): k for idx, k in enumerate(np.mean(Resources.getWordVectors().vectorize(words, remove_oov=True), axis=0))}
          
if __name__ == '__main__':
    cfg.makeFolders()
    
    umls = UMLSFeatures()
    #texts = utils.readData(cfg.PATH_INPUT)
    #for text in texts:
    #    texts[text]["tokens"] = utils.dumbTokeniser(texts[text]["note"])
    '''data = utils.readData(cfg.PATH_INPUT,cfg.PATH_PREPROCESSED,10)
    
    matcher = DictionaryMatcher(4,0.65)
    matcher = matcher.loadModel('FILEUMLS')
    
    #pickle.dump(data,open(cfg.PATH_TEMP+'dataDump.p','wb'))
    #data = pickle.load(open(cfg.PATH_TEMP+'dataDump.p','rb'))
    umls = UMLSFeatures()
    
    for ide in data:
        text = data[ide].getTextObject()
        concepts = matcher.getListOfConcepts(text, True)
        #print(umls.getUMLSFeatures(concepts))
        print(umls.getPsychFeatures(concepts))
        
    '''