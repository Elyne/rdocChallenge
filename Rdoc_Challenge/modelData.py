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

from sklearn import cross_validation
from sklearn import naive_bayes, svm
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, precision_score, recall_score, f1_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.grid_search import GridSearchCV
from QandAFeatures import QandAFeatures
import FeatureSelection as fs
import SampleSelection as ss
from ExperimentSettings import ExperimentSettings
from copy import copy
from collections import OrderedDict
from collections import defaultdict

import numpy as np
import random
import utils
import os

from UMLSFeatures import UMLSFeatures
import config as cfg


def mean_absolute_error_official(gold, system):
    """
    Helper function for mean absolute error.

    :param gold:
    :param system:
    :return:
    """

    gold, system = np.array(gold), np.array(system)
    output_errors = np.average(np.abs(system - gold))
    return output_errors


def mae(y_true, y_pred):
    """
    Function which computes the Macro-averaged Mean Absolute Error (MAE), normalize
    it wrt the highest possible error and convert it into a percentage
    score. The score ranges between 0 and 1:
     - 000: lowest score;
     - 100: highest score;
    """
    
    y_true = list(y_true)
    y_pred = list(y_pred)
    mae_per_score = []
    stats_per_score = dict()
    for score in set(y_true):
        result = 0
            
        # Filters gold and system, by looking at gold's elements.
        x_n, y_n = zip(*[(a, b) for a, b in zip(y_true, y_pred) if a == score])

        # According to the gold standard, I compute the sum of the maximum
        # error for each prediction.
        # In a scale (0, 1, 2, 3), the points 1 and 2 can lead to maximum
        # error 2. The points 0 and 3 can lead to maximum error 3.
        if score in (0, 3):
            normalisation_factor = 3
        else:
            normalisation_factor = 2
        # Compute micro-averaged MAE
        try:
            result = 100 * (1 - (mean_absolute_error_official(x_n, y_n) /
                                 normalisation_factor))
        except ValueError:
            # The system hasn't predicted anything with this score!
            result = 0

        mae_per_score.append(result)
        stats_per_score[score] = (len(x_n),
                                  y_pred.count(score),
                                  result)
    score = sum(mae_per_score) / len(set(y_true))
    return stats_per_score, score

''' 
This function creates a list of the unique features you want to extract
'''   
def getFeatVocab(data, featBins):

    vocab = set()
    for bin2 in featBins:
        for dataObj in data:
            try:
                feats = dataObj.getFeats()[bin2]
                for feat in feats:
                    vocab.add(feat)
            except KeyError:
                continue

    outVocab = dict()
    for i, key in enumerate(vocab):
        outVocab[key] = i
    return outVocab

def generateRandomSampleOrder(data):
    data_order = list(data.keys())
    random.shuffle(data_order)
    return data_order

def getFeats(data, featList, vocab=None):
    """
    get feature values from data for training ML model 
    @param data: dictionary, where each key is a file name and each value is a py object
    @param data_order: the list of elements you want to extract for this set
    @param featList: list of features to use
    @param vocab: standardly set to None. If something is here, it will use that vocab instead of generating a new one from file
    @return features: the feature values generated from note
    @return vocab: for subsequent sample processing (e.g. test), you want the vocab created here
    """
    if vocab is None:
        vocab = getFeatVocab(data, featList)

    features = np.zeros((len(data), len(vocab)), dtype=np.int) # shape: (nb_samples,nb_feats)
    
    for i, d in enumerate(data):
        for feat in featList:
            try:
                featVals = d.feats[feat]
                for key, val in featVals.items():
                    try:
                        features[i, vocab[key]] = val
                    except KeyError:
                        continue
            except KeyError:
                continue
                
    return features, vocab

def getWeights(data, data_order, useAnnotatorWeighing=False):
    weights = []
    for idx in data_order:
        if (useAnnotatorWeighing):
            anns = data[idx].Nannotators
            if anns == '3':
                weights.append(3)
            elif anns == '1':
                weights.append(2)
            elif anns == '2':
                weights.append(4)
            else:
                weights.append(1)
        else:
            weights.append(1)
    return weights

def generatePrimaryFeats(data, expSet):
    """
    Generate all kinds of features that can be generated before cross-validation
    
    IMPORTANT: if your feature derives anything from data samples other than the sample it is currently on,
    you NEED to add it to generateDataDrivenFeats to avoid any pollution from the test set
    
    """
    expSet.featTypes = [x.upper() for x in expSet.featTypes]
    umlsfeats = UMLSFeatures(expSet)

    for cnt, d in enumerate(data):
        if cnt % 50 == 0:
            print("Generated features for", cnt, "of", len(data), "samples.")

        d.generatePrimaryFeatures(expSet, umlsfeats)
        
def generateDataDrivenFeats(trainSet, data, expSet):
    featTypes = [x.upper() for x in expSet.featTypes]
    if set(featTypes).intersection({"BOW_ANSWERS", "CATEGORICAL_QUESTIONSET", "QUESTIONSET","PREAMBLE_CLUSTERS","CONCEPT_CLUSTERS", "LONG_QUESTIONSET", "CONCEPTS_FROM_SUMMARY"}):
        qandaFeats = QandAFeatures(expSet, trainSet)
        for _, d in enumerate(data):
            featBins = qandaFeats.apply(d)
            for bin2 in featBins:
                d.feats[bin2] = featBins[bin2]
                

def getScores(labels_true, labels_pred):
    
    print("Precision: ", precision_score(labels_true, labels_pred, average='weighted'))
    print("Recall: ", recall_score(labels_true, labels_pred, average='weighted'))
    print("F1 score: ", f1_score(labels_true, labels_pred, average='weighted'))
    print("Accuracy score: ", accuracy_score(labels_true, labels_pred))

    print("Mean absolute error (sklearn) on the test set is:", mean_absolute_error(labels_true, labels_pred))
    print("Mean absolute error (official) statistics (per score) and score on the test set is:", mae(labels_true, labels_pred))

    
def splitData(features, labels, testSize = 0.3):
    '''
    Split data into train and test sets
    @param features: Features generated from data
    @param labels: symptom severity label for each note
    @param testSize: fraction of data to use for testing models
    @return feats_train: the features for training
    @return feats_test: the features for testing
    @return labels_train: symptom severity labels corresponding to training features
    @return labels_test: symptom severity labels corresponding to test features 
    '''
    
    feats_train, feats_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=testSize, random_state=15)

    return(feats_train, feats_test, labels_train, labels_test)

def getEstimator(es):
    
    estimator = None
    algo = es.ml_algorithm.upper()
    if algo == 'NAIVEBAYESGAUSSIAN':
        estimator = naive_bayes.GaussianNB()
    elif algo == 'SVM':
        estimator = svm.SVC(kernel=es.svmKernel, degree = 3, C = 0.1, random_state=es.random_seed)
    elif algo == 'RF':
        estimator = RandomForestClassifier(n_estimators=100, random_state=es.random_seed)
    elif algo == 'DECISIONTREE':
        estimator = DecisionTreeClassifier(random_state=es.random_seed)
    elif algo == 'RANDOM':
        estimator = DummyClassifier(random_state=es.random_seed)
    else:
        print("Please enter correct estimator (NaiveBayesGaussian/SVM/RF/DecisionTree)")
        
    #TODO: add regression?
    return estimator

def conceptPreprocessing(data, removeDeniedConcepts, splitDeniedConcepts, removeUncertainConcepts, splitUncertainConcepts, removeFamilyHistory, splitFamilyHistory):
    """
    Preprocessing the present concepts in the list so that denied concepts can be modified in several ways.
    If conflicting settings are found here, the one appearing first in the parameter list will win.    
    """
    for d in data:
        if (removeDeniedConcepts):
            d.getTextObject().remove_concepts_from_denied_questions()
        elif (splitDeniedConcepts):
            d.getTextObject().separate_concepts_from_denied_questions()
        if (removeUncertainConcepts):
            d.getTextObject().remove_concepts_from_uncertain_questions()
        elif (splitUncertainConcepts):
            d.getTextObject().separate_concepts_from_uncertain_questions()
        if (removeFamilyHistory):
            d.getTextObject().remove_concepts_from_family_questions()
        elif (splitFamilyHistory):
            d.getTextObject().separate_concepts_from_family_questions()
    return data


def grid_search(estimator, data, featTypes=('BoW',), nFolds=10, random_seed=44, param_grid=()):

    labels = [x.severity for x in data]

    generatePrimaryFeats(data, featTypes)

    featurized = []
    for d in data:
        instance = {}
        for featname, values in d.feats.items():
            # Give each feature a unique name to avoid overwriting features.
            # If e.g. a concept feature has the same name as a bow word, the old code
            # would overwrite one of the features.
            instance.update({"{0}-{1}".format(featname, k): v for k, v in values.items()})

        featurized.append(instance)

    d = DictVectorizer()
    x_train = d.fit_transform(featurized)

    folds = cross_validation.StratifiedKFold(labels, n_folds=nFolds, shuffle=True, random_state=random_seed)
    grid = GridSearchCV(estimator, param_grid=param_grid, scoring="f1", n_jobs=-1, cv=folds)
    fit_grid = grid.fit(x_train, labels)

    print(fit_grid.best_params_)
    return fit_grid.best_params_ 

def get_estimators_params_rbf(k, random_seed = 44):
    estimators = []
    
    c_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    for c in c_range:
        for gamma in gamma_range:
            estimators.append((svm.SVC(kernel=k, C=c, gamma = gamma, degree=3, random_state=random_seed), c, gamma)) #(estimator, c, gamma)

    return estimators

def get_estimators_c_param(k, random_seed = 44):

    estimators = []
    
    c_range = np.logspace(-2, 10, 13)
    for c in c_range:
        estimators.append((svm.SVC(kernel=k, C=c, gamma = 'auto', random_state=random_seed), c, 'auto')) #(estimator, c, gamma)

    return estimators

def featurize(data):
    '''Calculate all features once,
     The vectorizer takes care of the separation between test and train.
    '''
    
    featurized = []
    for d in data:
        instance = {}
        for featname, values in d.feats.items():
        # Give each feature a unique name to avoid overwriting features.
        # If e.g. a concept feature has the same name as a bow word, the old code
        # would overwrite one of the features.
            instance.update({"{0}-{1}".format(featname, k): v for k, v in values.items()})
        
        featurized.append(instance)
                
    return featurized

def calc_and_append_scores(y_test, y_pred, metrics, featImportance):
    
    metrics['scores_mae'].append(mean_absolute_error(y_test, y_pred))
    _, score_off = mae(y_test, y_pred)
    metrics['scores_mae_official'].append(score_off)
    prec, rec, fmeasure, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
            
    metrics['scores_prec'].append(prec)
    metrics['scores_recall'].append(rec)
    metrics['scores_f1'].append(fmeasure)
    metrics['scores_accuracy'].append(accuracy_score(y_test, y_pred))
    metrics['feature_importance'].append(featImportance)
            
            
    # Getting class-individual metrics
    tTP = [0,0,0,0]
    tFP = [0,0,0,0]
    tTN = [0,0,0,0]
    tFN = [0,0,0,0]
            
    for act, pred in zip(y_test, y_pred):
        if act == pred:
            for i in range(0,4):
                if i == act: #add to true positive
                    tTP[i] += 1
                else: #add to true negative
                    tTN[i] += 1
        else:
            for i in range(0,4):
                if i == act: #add to false negative
                    tFN[i] += 1
                else: #add to false positive
                    tFP[i] += 1
            
    tpre = [0,0,0,0]
    trec = [0,0,0,0]
    tfm = [0,0,0,0]
    ttp = [0,0,0,0]
    for i in range(0,4):
        if (tTP[i] > 0.):
            tpre[i] = tTP[i] / (tTP[i] + tFP[i])
            trec[i] = tTP[i] / (tTP[i] + tFN[i])
        if ((trec[i] > 0.) | (tpre[i] > 0.)):
            tfm[i] = (2*(tpre[i] * trec[i])) / (tpre[i]+trec[i])
        ttp[i] = tTP[i]
        
    #for each label separately,
    # to see how well our model performs on separate labels
    metrics['indRec'].append(trec)
    metrics['indPrec'].append(tpre)
    metrics['indFmeasure'].append(tfm)
    metrics['indTP'].append(ttp)

def save_results(vectorizer, metrics, confMat, es, nFolds) :
    
    results = OrderedDict()
    results["Used features"] = str(len(vectorizer.get_feature_names()))
    results["Average Mean absolute error (sklearn)"] = np.average(metrics['scores_mae'])
    results["Average Mean absolute error (sklearn); std"] = np.std(metrics['scores_mae'])
    results["Average Mean absolute error (official)"] = np.average(metrics['scores_mae_official'])
    results["Average Mean absolute error (official); std"] = np.std(metrics['scores_mae_official'])
    results["Average Precision"] = np.average(metrics['scores_prec'])
    results["Average Precision; std"] = np.std(metrics['scores_prec'])
    results["Average Recall"] = np.average(metrics['scores_recall'])
    results["Average Recall; std"] = np.std(metrics['scores_recall'])
    results["Average F1-measure"] = np.average(metrics['scores_f1'])
    results["Average F1-measure; std"] = np.std(metrics['scores_f1'])
    results["Average Accuracy"] = np.average(metrics['scores_accuracy'])
    results["Average Accuracy; std"] = np.std(metrics['scores_accuracy'])
    results["Confusion matrix"] = confMat/nFolds
        
    for i in range(0,4):
        prec = str(round(np.average([x[i] for x in metrics['indPrec']]),3))
        rec = str(round(np.average([x[i] for x in metrics['indRec']]),3))
        fm = str(round(np.average([x[i] for x in metrics['indFmeasure']]),3))
        tp = str(round(np.average([x[i] for x in metrics['indTP']]),3))
        results["Class" + str(i)] = "Prec:" + prec +"; Rec:" + rec +"; F-measure:" + fm +"; TP:" +tp
        
        
    # Constructing feature importance matrix (top 50 features retained)
    sums = dict()
    counts = dict()
    try:
        for itemset in metrics['feature_importance']:
            for pair in itemset:
                try:
                    a = sums[pair[0]] + pair[1]
                    b = counts[pair[0]] + 1
                except:
                    a = pair[1]
                    b = 1
                sums[pair[0]] = a
                counts[pair[0]] = b  
            coll = [[x, float(sums[x])/counts[x]] for x in sums.keys()]
            indices = np.argsort([x[1] for x in coll])[::-1]
            results['FeatureImportance'] = [coll[x] for cn, x in enumerate(indices) if cn < 50]
    except:
        pass
        
    results["Markdown"] = ', '.join(es.featTypes) + '|' + es.ml_algorithm + '|'+ str(round(np.average(metrics['scores_mae_official']),1)) + '|' + str(round(np.average(metrics['scores_mae']),3))+ '|'+str(round(np.average(metrics['scores_prec']),3)) +'|' + str(round(np.average(metrics['scores_recall']),3)) +'|'+  str(round(np.average(metrics['scores_f1']),3)) + '|' + str(round(np.average(metrics['scores_accuracy']),3)) +'|'
        
    return results

def save_decision_tree(treePath, model, fold_idx, featNames):
    if not os.path.exists(treePath):
        os.makedirs(treePath)
    export_graphviz(model, out_file=treePath+'fold'+str(fold_idx)+'.dot', feature_names=featNames, filled=True, class_names=["absent","mild","moderate","severe"], proportion = True)

def param_sweep_svm(data, es, gammaSweep = False, nFolds = 10, verbose = False, random_seed = 44):
    
    result_params = dict() #{(kernel,c,gamma):result}
    
    svmKernels = ['rbf','linear','poly','sigmoid']
    for cur_kernel in svmKernels:
    
        if es.ml_algorithm == 'SVM':
            if gammaSweep and cur_kernel == 'rbf' or cur_kernel == 'poly' or cur_kernel == 'sigmoid':
                estimators = get_estimators_params_rbf(cur_kernel, random_seed)
            else:
                estimators = get_estimators_c_param(cur_kernel, random_seed)
        else:
            print("Parameter Sweep supported only for SVM.")
            return None
                    
    for cur_est, cur_c, cur_gamma in estimators:
        result_params[(cur_kernel, cur_c, cur_gamma)] = evalCrossval(cur_est, data, es, nFolds, printTree = False, verbose = verbose, random_seed = random_seed)
    
    return result_params

def get_bootstrapped_trainset(trainSet, y_train, bootstrap_data, es, estimator, th_bs):
    new_train_set = trainSet
    new_y_train = y_train
    
    trainAndBSData = trainSet + bootstrap_data
    
    generateDataDrivenFeats(trainSet, trainAndBSData, es)
    
    featurized = featurize(trainAndBSData)

    train_feats = [featurized[idx] for idx in range(0, len(trainSet), 1)]
    test_feats = [featurized[idx] for idx in range(len(trainSet), len(trainAndBSData), 1)]
        
    #Do feature selection on train data
    train_feats = fs.runFeatureSelection(train_feats, y_train, es)
    train_feats, y_train, train_bucket = ss.runSampleSelection(train_feats, y_train,[i for i in range(0, len(trainSet), 1)], es)
    
    # calculate Inter-annotator weighting. 
    weights_train = getWeights(trainAndBSData, train_bucket, es.weighInterAnnot)
    
    vectorizer = DictVectorizer()   
    x_train = vectorizer.fit_transform(train_feats)
    x_test = vectorizer.transform(test_feats)
        
    if es.scaleData:
        min_max_scalar = MinMaxScaler()
        x_train = min_max_scalar.fit_transform(x_train.toarray())
        x_test = min_max_scalar.transform(x_test.toarray())
        
    model = train(estimator, x_train, y_train, weights_train, model=None)
        
    y_pred_prob = model.predict_proba(x_test)
    for i, cur_y in enumerate(y_pred_prob):
        if np.max(cur_y) > th_bs:
            new_train_set.append(bootstrap_data[i])
            new_y_train.append(np.argmax(cur_y))
            
    return (new_train_set, new_y_train) #update none to confidence vector
    
    
def eval_bootstrapped_crossVal(estimator, data, bootstrap_data, es=ExperimentSettings(), nFolds = 10, printTree=False, verbose=False, th_bs = 0.6, random_seed = 44):
    labels = [x.severity for x in data]
    folds = cross_validation.StratifiedKFold(labels, n_folds=nFolds, shuffle=True, random_state=es.random_seed)
    
    min_max_scalar = MinMaxScaler()
    
    metrics = defaultdict(list)
    confMat = None
    
    generatePrimaryFeats(data, es)
    generatePrimaryFeats(bootstrap_data, es)
    utils.out('Generated primary features!')
     
    # For each fold
    for fold_idx, fold in enumerate(folds):
        #making an 'inner data' set, in which we have a copy of the original data (makes sure we do not modify the original data
        trainAndTestData = copy(data)
    
        train_bucket, test_bucket = fold
            
        # Generate data-driven features (meta-features)
        # These features should be generated within the loop, because some clustering might happen between samples (e.g. to determine which questions are 'regular')
        trainData = [copy(trainAndTestData[idx]) for idx in train_bucket]
        y_train = [labels[idx] for idx in train_bucket]
        
        (new_train_data, new_y_train) = get_bootstrapped_trainset(trainData, y_train, bootstrap_data, es, estimator, th_bs)
        
        testData = [copy(trainAndTestData[idx]) for idx in test_bucket]
        allData = new_train_data+testData
        generateDataDrivenFeats(new_train_data, allData, es)
            
        if verbose:
            utils.out('Generated data-driven features!')
            
        # Deriving the values for the trainset, also generating the vocabulary
        featurized = featurize(allData)
        # Get all featurized documents from by using the indices in the train and test buckets.
    
        train_feats = featurized[0:len(new_train_data)]
        test_feats = featurized[len(new_train_data):len(featurized)]
        
        #Do feature selection on train data
        
        train_feats = fs.runFeatureSelection(train_feats, new_y_train, es)
        train_feats, new_y_train, new_train_bucket = ss.runSampleSelection(train_feats, new_y_train,[i for i in range(len(new_train_data))], es)
            
        vectorizer = DictVectorizer()
        # Fit and transform the train data.        
        x_train = vectorizer.fit_transform(train_feats)
        # Same for test data.
        x_test = vectorizer.transform(test_feats)
        y_test = [labels[idx] for idx in test_bucket]
        
        new_weights_train = getWeights(new_train_data, new_train_bucket, es.weighInterAnnot)
        
        if es.scaleData:
            x_train = min_max_scalar.fit_transform(x_train.toarray())
            x_test = min_max_scalar.transform(x_test.toarray())
            
            
        if verbose:
            utils.out("Running fold", fold_idx)
    
        model = train(estimator, x_train, new_y_train, new_weights_train, model=None)
        # output the importance of features
        indices = np.argsort(model.feature_importances_)[::-1]
        featImportance = [[vectorizer.feature_names_[x], model.feature_importances_[x]] for x in indices]
        
        y_pred = test(x_test, model)
            
        if confMat is None:
            confMat = confusion_matrix(y_test, y_pred, [0,1,2,3])
        else:
            confMat += confusion_matrix(y_test, y_pred, [0,1,2,3])
            
        if verbose:
            utils.out("Actual", y_test)
            utils.out("Predicted", y_pred)
            
        if printTree:
            save_decision_tree(cfg.PATH_DECISION_TREE+'_'.join(es.featTypes)+"/", model, fold_idx, vectorizer.get_feature_names())
        
        calc_and_append_scores(y_test, y_pred, metrics, featImportance)
        
    return save_results(vectorizer, metrics, confMat, es, nFolds)
        
def evalCrossval(estimator, data, es=ExperimentSettings(), nFolds = 10, printTree=False, verbose=False, random_seed = 44):
    '''
    Calculate average cross validation score on the split train data to evaluate performance of trained models
    @param estimator: the machine learning estimator
    @param feats_train: Features for generated training data
    @param labels_train: symptom severity label for generated training data 
    @param nFolds: number of folds in k-fold cross validation 
    '''

    # scores = cross_validation.cross_val_score(estimator, feats_train, labels_train, scoring='mean_absolute_error', cv=nFolds, verbose=1)
    # print("Average cross validation score (mean absolute error): ", np.average(scores))
    
    labels = [x.severity for x in data]
    folds = cross_validation.StratifiedKFold(labels, n_folds=nFolds, shuffle=True, random_state=es.random_seed)
    
    min_max_scalar = MinMaxScaler()

    
    metrics = defaultdict(list)
    confMat = None
    
    generatePrimaryFeats(data, es)
    utils.out('Generated primary features!')

    #_, vocab = getFeats(data, ['MED'])
    #print(vocab)
     
    # For each fold
    for fold_idx, fold in enumerate(folds):
        #making an 'inner data' set, in which we have a copy of the original data (makes sure we do not modify the original data
        innerData = copy(data)
    
        train_bucket, test_bucket = fold
            
        # Generate data-driven features (meta-features)
        # These features should be generated within the loop, because some clustering might happen between samples (e.g. to determine which questions are 'regular')
        trainSet = [copy(innerData[idx]) for idx in train_bucket]
        generateDataDrivenFeats(trainSet, innerData, es)
            
        if verbose:
            utils.out('Generated data-driven features!')
            
        # Deriving the values for the trainset, also generating the vocabulary
        featurized = featurize(innerData)
        # Get all featurized documents from by using the indices in the train and test buckets.
        train_feats = [featurized[idx] for idx in train_bucket]
        test_feats = [featurized[idx] for idx in test_bucket]
            
        #Do feature selection on train data
        y_train = [labels[idx] for idx in train_bucket]
        train_feats = fs.runFeatureSelection(train_feats, y_train, es)
        train_feats, y_train, train_bucket = ss.runSampleSelection(train_feats, y_train,train_bucket, es)
        
            
        vectorizer = DictVectorizer()
        # Fit and transform the train data.        
        x_train = vectorizer.fit_transform(train_feats)
        # Same for test data.
        x_test = vectorizer.transform(test_feats)
        y_test = [labels[idx] for idx in test_bucket]
            
        if es.scaleData:
            x_train = min_max_scalar.fit_transform(x_train.toarray())
            x_test = min_max_scalar.transform(x_test.toarray())
            
        # calculate Inter-annotator weighting.
        weights_train = getWeights(data, train_bucket, es.weighInterAnnot)
            
        if verbose:
            utils.out("Running fold", fold_idx)        
        
        model = train(estimator, x_train, y_train, weights_train, model=None)
        
        
        #for part in model.estimators_:
            #graph = export_graphviz(part, out_file=None, feature_names=vectorizer.feature_names_)
            #selFeats = utils.find_between(graph, 'label="','gini')
            
        # output the importance of features
        try:
            indices = np.argsort(model.feature_importances_)[::-1]
            featImportances = [[vectorizer.feature_names_[x], model.feature_importances_[x]] for x in indices]    
        except:
            featImportances = None 
        
        y_pred = test(x_test, model)
        #print(y_pred)
            
        if confMat is None:
            confMat = confusion_matrix(y_test, y_pred, [0,1,2,3])
        else:
            confMat += confusion_matrix(y_test, y_pred, [0,1,2,3])
            
        if verbose:
            utils.out("Actual", y_test)
            utils.out("Predicted", y_pred)
            
        if printTree:
            save_decision_tree(cfg.PATH_DECISION_TREE+'_'.join(es.featTypes)+"/", model, fold_idx, vectorizer.get_feature_names())
        
        calc_and_append_scores(y_test, y_pred, metrics, featImportances)
        
    return save_results(vectorizer, metrics, confMat, es, nFolds)    

def train(estimator, feats_train, labels_train, weights_train, model='model.pkl'):
    '''
    Train and Evaluate (using k-fold cross validation) the generated machine learning model for severity classification
    @param estimator: the ML estimator to use
    @param feats_train: feats_train: the training features
    @param labels_train: labels for training data
    @return estimator: trained estimator (model)
    '''
    
    estimator = estimator.fit(feats_train, labels_train, sample_weight=weights_train)
    if model is not None:
        joblib.dump(estimator, cfg.PATH_RESOURCES+model)
    return estimator
    

def test(feats_test, estimator=None, model='model.pkl'):
    """
    Evaluate the generated machine learning model on test data, and print a mean absolute error.
    @param estimator: The trained ML model/estimator
    @param feats_test: test features (obtained from data)
    """
    if estimator is None:
        estimator = joblib.load(cfg.PATH_RESOURCES+model)
        
    return estimator.predict(feats_test)

