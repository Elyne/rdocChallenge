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


Created on 4 Aug 2016

@author: elyne
'''

from reach import Reach
import config as cfg
from utilIterators import FileIterator
from collections import defaultdict

class Resources(object):
    '''
    This class contains various resources, used in parts of the code.
    The purpose here is that you only need to load (bigger) resources once, and can share them over multiple components
    
    When calling for a resource, it initialises that resource if it is present, otherwise it gets the predefined
    These objects should be read-only!
    '''
    
    wordVectors = None
    conceptVectors = None
    dsmFilter = None
    dsmDev1Filter = None
    medFilter = None
    snomedFilter = None
    
    lexiconFilters = dict()
    
    @classmethod  
    def getWordVectors(self):
        '''
        Wraps a word vector file from reach, so you do not have to declare this in each feature generation file
        '''
        if self.wordVectors == None:
            self.wordVectors = Reach(cfg.PATH_RESOURCES+'psych_vectors.txt', header=True)
        return self.wordVectors

    @classmethod
    def getConceptVectors_dsm(self):
        '''
        Wraps a concept vector file from reach, so you do not have to declare this in each feature generation file
        '''
        if self.conceptVectors == None:
            self.conceptVectors = Reach(cfg.PATH_RESOURCES+'concepts_dsm.txt', header=False)
        return self.conceptVectors

    @classmethod
    def getConceptVectors_choi(self):
        '''
        Wraps a concept vector file from reach, so you do not have to declare this in each feature generation file
        '''
        if self.conceptVectors == None:
            self.conceptVectors = Reach(cfg.PATH_RESOURCES + 'concepts_choi.txt', header=False)
        return self.conceptVectors
    
    
    @classmethod
    def getLexiconFilter(self, lexiconName):
        try:
            lib = self.lexiconFilters[lexiconName]
        except:
            #make the new lib
            lib = dict()
            if (lexiconName == 'DSM'):
                for line in FileIterator(cfg.PATH_RESOURCES+'dsm4.txt'):
                    lib[line[0].split('-')[0]] = line[1][0]
            elif (lexiconName == 'DSM+1'):
                for line in FileIterator(cfg.PATH_RESOURCES+'dsm4_rel1.txt'):
                    lib[line[0].split('-')[0]] = line[1][0]
            elif (lexiconName == 'DSM+2'):
                for line in FileIterator(cfg.PATH_RESOURCES+'dsm4_rel2.txt'):
                    lib[line[0].split('-')[0]] = line[1][0]
            elif (lexiconName == 'SNOMED'):
                for line in FileIterator(cfg.PATH_RESOURCES+'SNOMED_PSYCH.txt'):
                    lib[line[0].split('-')[0]] = line[1][0]
            elif (lexiconName == 'SNOMED+1'):
                for line in FileIterator(cfg.PATH_RESOURCES+'SNOMED_PSYCH_rel1.txt'):
                    lib[line[0].split('-')[0]] = line[1][0]
            print(lexiconName, 'has', len(lib), 'records.')
            self.lexiconFilters[lexiconName] = lib
        return lib
    
    @classmethod
    def getMedFilter(self):
        if self.medFilter == None:
            self.medFilter = dict() #{RxNormCUI:RxNormString}
            self.activeCompFilter = dict() #{ATC_CUI:ATC_String}
            self.medRelFilter = defaultdict(list) #{RxNormCUI:[ACT_CUI]}
            for line in FileIterator(cfg.PATH_RESOURCES+'all_rxnorm.txt', splitTokens=False):
                cui1 = line[0].split('-')[0]
                self.medFilter[cui1] = line[1]
                cui2 = line[2].split('-')[0]
                self.activeCompFilter[cui2] = line[3]
                self.medRelFilter[cui1].append(cui2)
        return self.medFilter, self.activeCompFilter, self.medRelFilter
    
    '''@classmethod
    def getSnomedFilter(self):
        if self.snomedFilter == None:    
            self.snomedFilter = dict()
            #for line in FileIterator(cfg.PATH_RESOURCES+'dsm4_snomed_compound.txt'):
            for line in FileIterator(cfg.PATH_RESOURCES+'SNOMED_PSYCH.txt'):
                self.snomedFilter[line[0].split('-')[0]] = line[1][0]
        return self.snomedFilter
    
    @classmethod
    def getDSMdev1(self):
        if self.dsmDev1Filter == None:
            self.dsmDev1Filter = dict()
            for line in FileIterator(cfg.PATH_RESOURCES+'dsm4_rel1.txt'):
                self.dsmDev1Filter[line[0].split('-')[0]] = line[1][0]
        return self.dsmDev1Filter
    
    @classmethod
    def getDSM(self):
        if self.dsmFilter == None:
            self.dsmFilter = dict()
            for line in FileIterator(cfg.PATH_RESOURCES+'dsm4.txt'):
                self.dsmFilter[line[0].split('-')[0]] = line[1][0]
        return self.dsmFilter'''
    
        