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

def main(useAnnotatorWeighing=True):
    '''
    This script evaluates the system performance for the different runs on testdata, created with testSetup.py
    
    '''
    
    #runs = ['DSM+1,DIST_HIER','CONCEPTSwithoutContext','CONCEPTS+CONTEXT', 'BOW', 'DSM+2','CATEGORICAL_QUESTIONSET,QUESTIONSET,LONG_QUESTIONSET','DSM','SNOMED+1','DSM+1']
    runs = ['DSM+1']
    evaluateForFeats(runs)
    
def evaluateForFeats(feats):
    log = ''
    
    for run in feats:
        run = ','.join(run)
        data = utils.readData(cfg.PATH_OUTPUT+run + '/', cfg.PATH_PREPROCESSED_TEST)
        gs = utils.readData(cfg.PATH_TEST, cfg.PATH_PREPROCESSED_TEST)
        log += str([x.key for x in data]) + '\n'
        log += str([x.key for x in gs]) + '\n'
        
        labels = [x.severity for x in data]
        labels_gs = [x.severity for x in gs]
        log += str(labels) + '\n'
        log += str(labels_gs) + '\n'
        
        log += str("Scores for " + run + ": \n")
        log += m.getScores(labels_gs, labels)
        log += '\n\n'
    return log;
    
if __name__ == "__main__":
    main()
