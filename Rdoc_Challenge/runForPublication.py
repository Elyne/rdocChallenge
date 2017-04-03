'''
Created on 31 Mar 2017

@author: elyne
'''
import config as cfg
import utils
import modelData as m
import testSetup
import testEval
from ExperimentSettings import ExperimentSettings

def runExperiments(features, es, logFile):
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
        
        logFile.write('Executing for ' + ','.join(featTypes) + ' model.\n')
        es.featTypes = featTypes
        
        if es.svmParamSweep:
            result_params = m.param_sweep_svm(data, es, gammaSweep = False, nFolds = 10, verbose = False, random_seed = 44)
            for name in result_params:
                logFile.write(str(name)+ ": " + str(result_params[name])+'\n')
        else:
            estimator = m.getEstimator(es)
            if es.bootstrap:
                results = m.eval_bootstrapped_crossVal(estimator, data, bootstrap_data, es, 10, printTree=False)
            else:
                results = m.evalCrossval(estimator, data, es, 10, printTree=False)
            for name in results:
                logFile.write(str(name)+ ": " + str(results[name])+'\n')


if __name__ == '__main__':
    ## Shared settings
    features = [["BOW"],["DSM+1"],["DSM"],["SNOMED"],["SNOMED+1"],["DSM+2"],["CONCEPTS"]]
    es = ExperimentSettings()
    es.fs_varianceFilter = True
    es.bootstrap = False
    es.ss_prototyping = False
    es.weighInterAnnot= False
    
    
    
    ##First, running all experiments without context
    es.removeDeniedConcepts=False
    es.splitUncertainConcepts=False
    es.splitFamilyConcepts=False
    fout = open(cfg.PATH_OUTPUT+'TenFold_Base.txt','w')
    runExperiments(features, es, fout)
    fout.write("*******\nNow processing test set\n*****\n\n")
    testSetup.runForExperimentSettings(features, es)
    fout.write(testEval.evaluateForFeats(features))
    fout.close()
    

    ##Then, running all experiments with context
    es.splitUncertainConcepts=True
    es.removeDeniedConcepts=True
    es.splitFamilyConcepts=True
    fout = open(cfg.PATH_OUTPUT+'TenFold_Context.txt','w')
    #runExperiments(features, es, fout)
    fout.write("*******\nNow processing test set\n*****\n\n")
    testSetup.runForExperimentSettings(features, es)
    fout.write(testEval.evaluateForFeats(features))
    fout.close()
    
    
    ##Then, running all experiments with context+outlierdetection+bootstrapping
    es = ExperimentSettings()
    es.fs_varianceFilter = True
    es.bootstrap = True
    es.ss_prototyping = True
    es.weighInterAnnot= False
    
    fout = open(cfg.PATH_OUTPUT+'TenFold_Context+Bootstrap+OutlierDetection.txt','w')
    runExperiments(features, es, fout)
    fout.write("*******\nNow processing test set\n*****\n\n")
    testSetup.runForExperimentSettings(features, es)
    fout.write(testEval.evaluateForFeats(features))
    fout.close()
    
    