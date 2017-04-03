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

import config as cfg
import utils
import modelData as m
import FeatureSelection as fs
import SampleSelection as ss
from ExperimentSettings import ExperimentSettings
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MinMaxScaler

import os

def main(useAnnotatorWeighing=True):
    '''
    This script runs the experiments by training on a trainset and testing on a test set. Also allows bootstrapping (which is hard coded in this script as well)
    Configure your model settings by modifying the ExperimentSettings object in the script.

    The output of these models are annotated files in the output folder, which can be evaluated (in metrics) using testEval.py
    '''

    # Making folders from config
    # cfg.makeFolders()

    # Here, you can specify the feature sets you would like to use. It is arranged in an array of arrays, to enable combinations
    features = [["DSM+1"]]
    #features = [["CONCEPTS"]]#['BOW'],
#     features = [["CONCEPTS"]]

    # if you want anything set differently than default, please change the corresponding parameter in es (ExperimentSettings)
    es = ExperimentSettings()
#     es.fs_varianceFilter = True
#     es.bootstrap = True
#     es.ss_prototyping = True
#     es.weighInterAnnot = False
#     es.ml_algorithm='RF'
    #remove these!
#     es.removeDeniedConcepts=False
#     es.splitFamilyConcepts=False
#     es.splitUncertainConcepts=False
    runForExperimentSettings(features, es)

def runForExperimentSettings(features, es):

    # Reading the train/test_data into an array
    train_data = utils.readData(cfg.PATH_TRAIN, cfg.PATH_PREPROCESSED_TRAIN)
    test_data = utils.readData(cfg.PATH_TEST, cfg.PATH_PREPROCESSED_TEST)

    # Doing modifications on the concepts, based on the segmentation settings that are defined (ONLY PERFORM ONCE)
    train_data = m.conceptPreprocessing(train_data, es.removeDeniedConcepts, es.splitDeniedConcepts, es.removeUncertainConcepts, es.splitUncertainConcepts, es.removeFamilyConcepts, es.splitFamilyConcepts)
    test_data = m.conceptPreprocessing(test_data, es.removeDeniedConcepts, es.splitDeniedConcepts, es.removeUncertainConcepts, es.splitUncertainConcepts, es.removeFamilyConcepts, es.splitFamilyConcepts)

    # Reading in bootstrap data as well when enabled
    if es.bootstrap:
        bootstrap_data = utils.readData(cfg.PATH_UNANNOTATED, cfg.PATH_PREPROCESSED_UNANNOTATED)
        bootstrap_data = m.conceptPreprocessing(bootstrap_data, es.removeDeniedConcepts, es.splitDeniedConcepts, es.removeUncertainConcepts, es.splitUncertainConcepts, es.removeFamilyConcepts, es.splitFamilyConcepts)

    # Doing modifications on the concepts, based on the segmentation settings that are defined (ONLY PERFORM ONCE)
    # train_data = m.conceptPreprocessing(train_data, es.removeDeniedConcepts, es.splitDeniedConcepts, es.removeUncertainConcepts, es.splitUncertainConcepts,es.removeFamilyConcepts,es.splitFamilyConcepts)
    # test_data = m.conceptPreprocessing(test_data, es.removeDeniedConcepts, es.splitDeniedConcepts, es.removeUncertainConcepts, es.splitUncertainConcepts,es.removeFamilyConcepts,es.splitFamilyConcepts)

    vectorizer = DictVectorizer()
    min_max_scalar = MinMaxScaler()

    # Looping over different feature parameters
    for featTypes in features:
        utils.out('Executing for ' + ','.join(featTypes) + ' model.')
        es.featTypes = featTypes

        estimator = m.getEstimator(es)

        m.generatePrimaryFeats(train_data, es)
        m.generatePrimaryFeats(test_data, es)
        utils.out('Generated primary features for train and test_data!')

        y_train = [d.severity for d in train_data]

        #else argument added here to not override the train_data/y_train setting, otherwise we can only do one featType at a time
        if es.bootstrap:
            m.generatePrimaryFeats(bootstrap_data, es)
            (train_datac, y_trainc) = m.get_bootstrapped_trainset(train_data, y_train, bootstrap_data, es, estimator, th_bs=0.6)
        else:
            train_datac = train_data
            y_trainc = y_train

        concatenated_data = []
        concatenated_data.extend(train_datac)
        concatenated_data.extend(test_data)

        m.generateDataDrivenFeats(train_datac, concatenated_data, es)

        featurized = m.featurize(concatenated_data)

        train_feats = featurized[0:len(train_datac)]
        test_feats = featurized[len(train_datac):len(featurized)]

        # Do feature selection on train data
        train_feats = fs.runFeatureSelection(train_feats, y_trainc, es)
        train_feats, y_trainc, train_bucket = ss.runSampleSelection(train_feats, y_trainc, [i for i in range(len(train_datac))], es)

        x_train = vectorizer.fit_transform(train_feats)
        x_test = vectorizer.transform(test_feats)

        if es.scaleData:
            x_train = min_max_scalar.fit_transform(x_train.toarray())
            x_test = min_max_scalar.transform(x_test.toarray())

        weights_train = m.getWeights(train_datac, train_bucket, es.weighInterAnnot)

        model = m.train(estimator, x_train, y_trainc, weights_train, model=None)

        y_pred = m.test(x_test, estimator=model)
#         print(y_pred)
        for i, cur_data in enumerate(test_data):
            cur_data.predSev = y_pred[i]

        out_dir = cfg.PATH_OUTPUT + ','.join(featTypes) + '/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        utils.genOutput(data=test_data, outDir=out_dir , dtd=cfg.PATH_OUTPUT + '2016_CEGS_N-GRID_TRACK2.dtd/')

if __name__ == "__main__":
    main()
