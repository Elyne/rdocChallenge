'''
Created on Jul 26, 2016

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

class ExperimentSettings(object):
    '''
    ExperimentSettings contains all possible settings you can make for running an experiment. It defines only one experiment!
    
    Also contains shared elements (such as the word_vector files)
    
    '''


    def __init__(self):
        # Features contains all sets of features that need to be derived
        self.featTypes = ["DSM+1"]
        # Options:
        # 'CONCEPTS', 'DSM+1', 'DSM', 'DSM_HIER', 'MED', 'BOW', 'BOW_ANSWERS', 'CATEGORICAL_QUESTIONSET', 'QUESTIONSET'
        # 'WORD_VECTOR', 'WORD_VECTOR_ANSWERS', 'CONCEPT_VECTOR', 'DIST_WORDVECTOR', 'DIST_CONCEPTVECTOR'
        # 'CONCEPT_CLUSTERS', 'PREAMBLE_CLUSTERS'
        
        # Specify the machine learning algorithm you want to use
        # Current options are: 'NaiveBayesGaussian', 'SVM', 'RF', 'DecisionTree'
        self.ml_algorithm = 'RF'
        
        #settings for svm kernel
        self.svmParamSweep = False
        self.svmKernel = 'poly'
        
        #Scaling the data (for SVM purposes)
        self.scaleData = False
        
        #bootstrap unannotated data or not
        self.bootstrap = False
        
        ##  Segmentation options
        # Remove concepts that are in 'denied' questions and answers
        self.removeDeniedConcepts = True
        # Modify concepts that are 'denied', meaning that they get a separate prefix (so you know they are denied)
        self.splitDeniedConcepts = False
        self.removeUncertainConcepts=False
        self.splitUncertainConcepts=True
        self.removeFamilyConcepts=False
        self.splitFamilyConcepts=True
        
        #Other settings
        self.random_seed = 44 # 44 for main run
        self.weighInterAnnot=False

        
        # Feature selection settings
        self.fs_chiSquare = False
        self.fs_chiSquareTopK = 180
        
        self.fs_varianceFilter = True
        self.fs_varianceThreshold = 0.95
        
        self.fs_confidence = False
        self.fs_confidenceWithCoverage = False
        self.fs_confidenceTopK = 180
        self.fs_confidenceCoverageOverlap = 3
        self.fs_confidenceCutOff = 0.05
        self.fs_confidenceValueDistinction = False
        
        self.fs_informationGain = False
        self.fs_informationGainWithCoverage = False
        self.fs_informationGainTopK = 250
        self.fs_informationGainCoverageOverlap = 3
        self.fs_informationGainCutOff = 0.05
        
        #sample selection settings
        self.ss_prototyping = False
        self.ss_prototyping_cutOff = 0.05
        