import config as cfg
import utils
import modelData as m
from ExperimentSettings import ExperimentSettings

"""
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
"""

def main(useAnnotatorWeighing=True):
    """
    This script allows for 10-fold cross validation over the data in the training set. Experiments only yield results, they don't yield annotated files.
    The standard deviation seen over the different folds for each metric are reported as well.
    
    Configure your model settings by modifying the ExperimentSettings object in the script.
    """
    
    # Making folders from config
    # cfg.makeFolders()
    
    # Here, you can specify the feature sets you would like to use. It is arranged in an array of arrays, to enable combinations
    features = [["DSM+1"]]
    #features = [["DSM"],["DSM+1","DIST_HIER"],["DSM+1"], ["CATEGORICAL_QUESTIONSET","QUESTIONSET","LONG_QUESTIONSET"]]
    
    # Options:
    # 'CONCEPTS', 'DSM+1', 'DSM', 'DSM_HIER', 'MED', 'BOW', 'BOW_ANSWERS', 'CATEGORICAL_QUESTIONSET', 'QUESTIONSET'
    # 'WORD_VECTOR', 'WORD_VECTOR_ANSWERS', 'CONCEPT_VECTOR', 'DIST_WORDVECTOR', 'DIST_CONCEPTVECTOR'
    # 'CONCEPT_CLUSTERS', 'PREAMBLE_CLUSTERS'
    
    # if you want anything set differently than default, please change the corresponding parameter in es (ExperimentSettings)
    es = ExperimentSettings()
    es.fs_varianceFilter = True
    es.bootstrap = False
    es.ss_prototyping = False
    es.weighInterAnnot= False
    #es.ml_algorithm='XGBOOST'
    #es.ml_algorithm = 'RANDOM'
    
    '''es.removeDeniedConcepts=True
    es.removeUncertainConcepts=False
    es.splitDeniedConcepts=False
    es.splitFamilyConcepts=True'''

    #es.fs_confidence=True
    #es.fs_confidenceValueDistinction = True
    #es.fs_chiSquare = False
    #es.fs_varianceFilter = True
    #es.fs_varianceThreshold = 0.05
    #es.fs_confidence = True 
    #es.fs_informationGain = False
    #es.fs_confidenceWithCoverage = True
    #es.fs_confidenceTopK = 100
    #es.fs_confidenceCoverageOverlap = 3
    #es.fs_confidenceCutOff = 0.05'''
    
    # Reading the data into an array
    data = utils.readData(cfg.PATH_TRAIN, cfg.PATH_PREPROCESSED_TRAIN)
    
    # Doing modifications on the concepts, based on the segmentation settings that are defined (ONLY PERFORM ONCE)
    data = m.conceptPreprocessing(data, es.removeDeniedConcepts, es.splitDeniedConcepts, es.removeUncertainConcepts, es.splitUncertainConcepts,es.removeFamilyConcepts,es.splitFamilyConcepts)
    
    if es.bootstrap:
        bootstrap_data = utils.readData(cfg.PATH_UNANNOTATED, cfg.PATH_PREPROCESSED_UNANNOTATED)
        bootstrap_data = m.conceptPreprocessing(bootstrap_data, es.removeDeniedConcepts, es.splitDeniedConcepts, es.removeUncertainConcepts, es.splitUncertainConcepts,es.removeFamilyConcepts,es.splitFamilyConcepts)      
    # Looping over different feature parameters
    for featTypes in features: 
        #for x in [True, False]:
            #es.fs_confidence = x
        
        utils.out('Executing for ' + ','.join(featTypes) + ' model.')
        es.featTypes = featTypes
        
        if es.svmParamSweep:
            result_params = m.param_sweep_svm(data, es, gammaSweep = False, nFolds = 10, verbose = False, random_seed = 44)
            for name in result_params:
                print(str(name)+ ":",result_params[name])
        else:
            estimator = m.getEstimator(es)
            if es.bootstrap:
                results = m.eval_bootstrapped_crossVal(estimator, data, bootstrap_data, es, 10, printTree=False)
            else:
                results = m.evalCrossval(estimator, data, es, 10, printTree=False)
            for name in results:
                print(str(name)+ ":",results[name])
     
if __name__ == "__main__":
    main()
