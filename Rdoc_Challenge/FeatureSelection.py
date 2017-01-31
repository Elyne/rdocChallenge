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

This model contains several feature selection algorithm.

- chisquare
- frequency filter
- variance filter
- confidence top-K/coverage
- information gain top-K/coverage

@author Elyne Scheurwegs


Created on Jul 25, 2016

'''
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold
from ExperimentSettings import ExperimentSettings
from operator import itemgetter
import math
from collections import Counter


def chiSquare(train_data, train_classes, topK):
    vectorizer = DictVectorizer()  
    
    # Fit and transform the train data.        
    x_train = vectorizer.fit_transform(train_data)
    y_train = train_classes
    
    if (x_train.shape[1] < topK):
        topK = x_train.shape[1]
    
    selector = SelectKBest(chi2, k=topK)
    x_new = selector.fit_transform(x_train, y_train)
    
    return vectorizer.inverse_transform(selector.inverse_transform(x_new))


def frequencyFilter(train_data, train_classes, threshold):
    '''
    Frequency filter
    '''
    #making a set of unique features (to be able to add the zero-case as well)
    ufs = dict()
    for feats in train_data:
        for feat in feats:
            if (feat != 0):
                try:
                    ufs[feat]+=1
                except:
                    ufs[feat] = 1
    
    thr = (threshold * (1 - threshold)) * len(train_data)
    sel = set()
    for feat in ufs:
        if ufs[feat] > thr:
            sel.add(feat)
            
    ntd = []
    for feats in train_data:
        sample = dict()
        for feat in feats:
            if feat in sel:
                sample[feat] = feats[feat]
        ntd.append(sample)
    return ntd


def varianceFilter(train_data, train_classes, threshold):
    #if True:
    #    return frequencyFilter(train_data, train_classes, threshold)
    '''
    Variance filter
    '''
    vectorizer = DictVectorizer()  
    # Fit and transform the train data.        
    x_train = vectorizer.fit_transform(train_data)
    #y_train = train_classes
    
    sel = VarianceThreshold(threshold=(threshold * (1 - threshold)))
    x_new = sel.fit_transform(x_train)
    return vectorizer.inverse_transform(sel.inverse_transform(x_new))

def confidenceScoring(train_data, train_classes, settings):
    '''
    Algorithm that calculates the confidence score for each occurring feature-class pair
    '''
    
    #first, we get all unique classes
    cls = set()
    for cl in train_classes:
        cls.add(cl)
    cls = list(cls)
    
    numbers = dict()
    for feats, cl in zip(train_data, train_classes):
        for feat in feats:
            if (feats[feat] != 0):
                try:
                    curr = numbers[feat]
                except:
                    curr = [0]
                    for i in range(0, len(cls)):
                        curr.append(0)
                curr[0] += 1
                curr[cls.index(cl)+1] += 1
                numbers[feat] = curr
    
    #now calculating confidence scores for each class
    res = []
    for infoOnFeat in numbers:
        curr = numbers[infoOnFeat]
        for i in range(1, len(curr)):
            curr[i] = curr[i] / curr[0]
        curr[0] = infoOnFeat
        res.append(curr)
        
    return res

def confidenceWithValueScoring(train_data, train_classes, es):
    '''
    Algorithm that calculates the confidence score for each occurring feature-class pair, but the feature also has a value linked to it.
    The max-confidence for a feature-value pair is taken as the confidence of that feature
    Uses the variance defined for the variance filter to determine the minimum number of samples within a value-group
    '''
    
    ##  Deciding the groups of values we want to make
    minSamplesInValueGroup = (es.fs_varianceThreshold * (1-es.fs_varianceThreshold)) * len(train_data)
    featVal = dict()
    featBins = dict()
    for feats in train_data:
        for feat in feats:
            try:
                lib = featVal[feat]
            except:
                lib = []
            lib.append(feats[feat])
            featVal[feat] = lib
            
    for feat in featVal:
        eligibleBins = [float('inf'), float('-inf')]
        bins = Counter(featVal[feat])
        for val in bins:
            if (bins[val] > minSamplesInValueGroup):
                eligibleBins.append(val)
        ##then, we extend the bins to have limits touching each other
        ##this is a very temporary solution! the binning should also take the distribution of values within into account (but it doesnt)
        eligibleBins = sorted(eligibleBins)
        eligibleBins = [(a + b) / 2 for a, b in zip(eligibleBins[:], eligibleBins[1:])]
        featBins[feat] = [((a, b)) for a, b in zip(eligibleBins[:], eligibleBins[1:])]
    
    ##here, we mutate the confidence algorithm so it uses multiple value pairs for each feature
    enriched_train_data = []
    for feats in train_data:
        sample = dict()
        for feat in feats:
            currBins = featBins[feat]
            #look in which bin the value falls, add that bin..
            val = feats[feat]
            nfName = feat + "-0"
            for i, bin2 in enumerate(currBins):
                if (val >= bin2[0]) & (val < bin2[1]):
                    nfName = feat + "-" + str(i)
                    break
            sample[nfName] = feats[feat]
        enriched_train_data.append(sample)

    #first, we get all unique classes
    cls = set()
    for cl in train_classes:
        cls.add(cl)
    cls = list(cls)
    
    numbers = dict()
    for feats, cl in zip(enriched_train_data, train_classes):
        for feat in feats:
            if (feats[feat] != 0):
                try:
                    curr = numbers[feat]
                except:
                    curr = [0]
                    for i in range(0, len(cls)):
                        curr.append(0)
                curr[0] += 1
                curr[cls.index(cl)+1] += 1
                numbers[feat] = curr
            else:
                print(feats[feat], feat)
    
    #now calculating confidence scores for each class
    res = []
    names = []
    for infoOnFeat in numbers:
        curr = numbers[infoOnFeat]
        for i in range(1, len(curr)):
            curr[i] = curr[i] / curr[0]
        curr[0] = infoOnFeat
        #go back to the original features with a max function
        split = infoOnFeat.split('-')
        name = '-'.join(split[0:len(split)-1])
        curr[0] = name
        
        if name in names:
            row = res[names.index(name)]
            for i in range(1, len(row)):
                if curr[i] > row[i]:
                    row[i] = curr[i]
        else:                
            res.append(curr)
            names.append(name)
            
    return res

def informationGainScoring(train_data, train_classes, settings):
    
    def entropy(set2, cls):
        entSum = 0
        for cl in cls:
            #count occurrences of cl
            tcl = set2.count(cl)
            if (tcl != 0):
                temp = (tcl/len(set2)) * math.log2(tcl/len(set2))
                entSum += temp
        return -entSum
    
    #first, we get all unique classes
    cls = set()
    for cl in train_classes:
        cls.add(cl)
    cls = list(cls)
    
    parentEntropy = entropy(train_classes, cls)
    
    #making a set of unique features (to be able to add the zero-case as well)
    ufs = set()
    for feats in train_data:
        for feat in feats:
            ufs.add(feat)
    
    #now, making a list of train_classes values for each value of feature x
    numbers = dict()
    for feats, cl in zip(train_data, train_classes):
        for feat in ufs:
            try:
                featVal = feats[feat]
            except:
                featVal = 0
            try:
                lib = numbers[feat]
            except:
                lib = dict()
            try:
                lib[featVal].append(cl)
            except:
                lib[featVal] = [cl]
            numbers[feat] = lib
    
    informationGain = []
    for feat in numbers:
        # calculating the information gain over all value subsets
        childEntropy = 0
        for childIde in numbers[feat]:
            child = numbers[feat][childIde]
            fraction = len(child) / len(train_classes)
            childEntropy += (fraction * entropy(child, cls))
        informationGain.append([feat, parentEntropy-childEntropy])
    return informationGain
    
def topK(train_data, train_classes, scoringFunc, settings = ExperimentSettings(), k=100):
    '''
    This function returns the top K features for each class. This means that the total number of features retained can exceed the limits
    '''    
    
    scores = scoringFunc(train_data, train_classes, settings)
    if (len(scores) < k):
        k = len(scores)
    selected = set()
    for i in range(1, len(scores[0])):
        scores = sorted(scores, key=itemgetter(i))
        for j, row in enumerate(scores):
            if (j < k):
                selected.add(row[0])
    
    print(len(selected), 'of', len(scores), ' unique features have been selected!')
    ntd = []
    for sample in train_data:
        new_feats = dict()
        for feat in sample:
            if feat in selected:
                new_feats[feat] = sample[feat]
        ntd.append(new_feats)
    return ntd

def coverage(train_data, train_classes, scoringFunc, settings = ExperimentSettings(), coverageOverlap=3, minimumUniqueSamples=3):
    '''
    This function returns features that maximally cover all samples of a certain class. It uses an ordered list, so it is 
    more likely to pick a higher scoring feature than a lower one
    '''
    coveredSamples = dict()
    featureToSampleMap = dict()
    selected = set()
    # first, we make a feature to sample map, so we don't have to iterate over all samples each time
    for ide, sample in enumerate(train_data):
        for feat in sample:
            if sample[feat] != 0:
                #using ide as sample identifier here
                try:
                    featureToSampleMap[feat].add(ide)
                except:
                    featureToSampleMap[feat] = set()
                    featureToSampleMap[feat].add(ide)
    
    scores = scoringFunc(train_data, train_classes, settings)
    for i in range(1, len(scores[0])):
        scores = sorted(scores, key=itemgetter(i))
        for score in scores:
            uniqueSamples = 0
            currSamples = featureToSampleMap[score[0]]
            for currSample in currSamples:
                try:
                    currCov = coveredSamples[currSample]
                    if currCov <= coverageOverlap:
                        uniqueSamples += 1
                    coveredSamples[currSample] = currCov + 1
                except:
                    uniqueSamples += 1
                    coveredSamples[currSample] = 1
            if uniqueSamples >= minimumUniqueSamples:
                selected.add(score[0])
                
    print(len(selected), 'of', len(scores), ' unique features have been selected!')
    
    ntd = []
    for sample in train_data:
        new_feats = dict()
        for feat in sample:
            if feat in selected:
                new_feats[feat] = sample[feat]
        ntd.append(new_feats)
    return ntd
    
    
def runFeatureSelection(train_data, train_classes, es=ExperimentSettings()):
    #confidenceScoring(train_data, train_classes)
    
    if es.fs_varianceFilter:
        train_data = varianceFilter(train_data, train_classes, es.fs_varianceThreshold)
    
    if es.fs_chiSquare:
        train_data = chiSquare(train_data, train_classes, es.fs_chiSquareTopK)
    
    if es.fs_confidence:
        if es.fs_confidenceValueDistinction == True:
            confFunc = confidenceWithValueScoring
        else:
            confFunc = confidenceScoring
        
        if es.fs_confidenceWithCoverage:
            train_data = coverage(train_data, train_classes, confFunc, settings=es, coverageOverlap=es.fs_confidenceCoverageOverlap)
        else:
            train_data = topK(train_data, train_classes, confFunc, settings = es, k=es.fs_confidenceTopK)
            
    if es.fs_informationGain:
        if es.fs_informationGainWithCoverage:
            train_data = coverage(train_data, train_classes, informationGainScoring, settings = es, coverageOverlap=es.fs_informationGainCoverageOverlap)
        else:
            train_data = topK(train_data, train_classes, informationGainScoring, settings = es, k=es.fs_informationGainTopK)
            
    #TODO: add information-gain coverage/topK

    return train_data
    