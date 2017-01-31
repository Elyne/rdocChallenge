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


Created on 5 Aug 2016

@author: elyne
'''
import numpy as np
from operator import itemgetter

def getPrototypicalSamples(train_data, train_classes, train_bucket, cutOff=0.05):
    """
    This function will select samples for each class that are most prototypical for that class.
    By performing this, we try to remove outliers, which could make for more robust models.
    
    :param train_data: the feature dictionaries of the samples
    :param train_classes: the classes of the samples
    :param cutoff: the percentage of least similar files to shave off
    :return: A tuple containing the filtered train_data and train_classes
    """
    
    def euclideanDistance(sample, others):
        distV = 0
        #using euclidean distance
        for oth in others:
            distV += np.linalg.norm(sample-oth)
        return distV / len(others)
    
    ##First, we make dense normalised (scale 0-1) vectors for all samples
    uf = dict()
    for feats in train_data:
        for feat in feats:
            try:
                if feats[feat] > uf[feat]:
                    uf[feat] = feats[feat]
            except:
                uf[feat] = feats[feat]
    cols = list(uf.keys())
                
    norm_data = []
    for feats in train_data:
        sample = []
        for col in cols:
            try:
                sample.append(feats[col]/uf[col])
            except:
                sample.append(0.)
        norm_data.append(np.array(sample))
    
    
    ##Then, we calculate how similar a sample is towards all other samples of this class a
    clSims = []
    for ide, sample in enumerate(norm_data):
        clSims.append(euclideanDistance(sample, [x for y, x in enumerate(norm_data) if train_classes[y] == train_classes[ide]]))
    
    ##Then the similarity between all other samples of other classes b
    oclSims = []
    for ide, sample in enumerate(norm_data):
        oclSims.append(euclideanDistance(sample, [x for y, x in enumerate(norm_data) if train_classes[y] != train_classes[ide]]))
        
    ##Get the x % of samples with the best ratio of a / b (for each class)
    scores = dict()
    for ide, sample in enumerate(norm_data):
        try:
            scores[train_classes[ide]].append([ide,clSims[ide]/oclSims[ide]])
        except:
            scores[train_classes[ide]] = [[ide,clSims[ide]/oclSims[ide]]]
    
    ret = set()
    for cl in scores:
        #sort scores[cl]
        scores[cl] = sorted(scores[cl], key=itemgetter(1))
        #selecting 
        thr = len(scores[cl]) * (1-cutOff)
        for i, s in enumerate(scores[cl]):
            if (i <= (thr-1)):
                ret.add(s[0])
    return ([s for ide, s in enumerate(train_data) if ide in ret], [s for ide, s in enumerate(train_classes) if ide in ret], [s for ide, s in enumerate(train_bucket) if ide in ret])
    
def runSampleSelection(train_feats, y_train, train_bucket, es):
    if es.ss_prototyping:
        train_feats, y_train, train_bucket = getPrototypicalSamples(train_feats, y_train, train_bucket, es.ss_prototyping_cutOff)
        
    return (train_feats, y_train, train_bucket)





