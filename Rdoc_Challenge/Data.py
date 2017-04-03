from collections import Counter
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
'''


class Data(object):
    
    def __init__(self, text, severity, Nannotators, key):
        self.text = text
        self.severity = severity
        self.feats = {}
        self.Nannotators = Nannotators
        self.predSev = None
        self.key = key
        
    def getTextObject(self):
        return self.text
    
    def getSeverity(self):
        return self.severity
    
    def getFeats(self):
        return self.feats
    
    def getTokenFreq(self):
        return Counter(self.text.get_tokens())

    def generatePrimaryFeatures(self, expSet, umlsfeats):
        """
        This function generates all PRIMARY features. That means that any data-driven features should be added in modelData/generateDataDrivenFeatures()
        
        """
        self.feats = dict()
        
        # declare umls before iterating
        if any(i in expSet.featTypes for i in ['CONCEPTS','DSM', 'DSM+1', 'DSM+2','SNOMED','SNOMED+1', 'DSM_HIER', 'MED', 'DIST_WORDVECTOR', 'DIST_CONCEPTVECTOR','SNOMED', 'CONCEPT_VECTOR']):
            concepts = self.text.get_concepts()
        
        for cur_feat in expSet.featTypes:
            if cur_feat == 'BOW':
                self.feats[cur_feat] = self.getTokenFreq()
            elif cur_feat == 'CONCEPTS':
                self.feats[cur_feat] = umlsfeats.getUMLSFeatures(concepts)
            elif cur_feat in ['DSM', 'DSM+1', 'DSM+2','SNOMED','SNOMED+1']:
                self.feats[cur_feat] = umlsfeats.getLexiconFeatures(concepts,cur_feat, True, True)
            elif cur_feat == 'MED':
                self.feats[cur_feat] = umlsfeats.getMedFeats(concepts, getMeds = True, getActComp = False, getMeta = True, verbose = True)
            elif cur_feat == "CONCEPT_VECTOR":
                self.feats[cur_feat] = umlsfeats.getVectorizedConcepts(concepts)
            elif cur_feat == "WORD_VECTOR_ANSWER":
                self.feats[cur_feat] = umlsfeats.getVectorizedWords(self.getTextObject().get_non_denied_text())
            elif cur_feat == "WORD_VECTOR":
                self.feats[cur_feat] = umlsfeats.getVectorizedWords(self.getTextObject().get_tokens())
            elif cur_feat == 'DIST_WORDVECTOR':
                self.feats[cur_feat] = umlsfeats.calculateAverageWordVectorDistance(concepts, filterName='ALL')
            elif cur_feat == 'DIST_CONCEPTVECTOR':
                self.feats[cur_feat] = umlsfeats.calculateAverageConceptVectorDistance(concepts, filterName='ALL')
            elif cur_feat == 'DIST_HIER':
                self.feats[cur_feat] = umlsfeats.getHierarchicalDistanceFeatures(concepts, filterName='ALL')
            
