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

Created on 3 Jun 2016
'''
import MySQLdb
import os

##Connection information
def connUMLS():
    return MySQLdb.connect(host='<your_hostname_here>',
                           port=3306,
                           user="<your username here>",
                           passwd="<your password here>",
                           db="umls2015")

##Various paths
PATH_TRAIN = '/home/elyne/git/rdocChallenge/input/train/'
PATH_PREPROCESSED_TRAIN = PATH_TRAIN + 'preprocessed/'

PATH_TEST = '/home/elyne/git/rdocChallenge/input/test/'
PATH_PREPROCESSED_TEST = PATH_TEST + 'preprocessed/'

PATH_RESOURCES = '/home/elyne/git/rdocChallenge/resources/libs/'
PATH_DECISION_TREE = '/home/elyne/git/rdocChallenge/resources/decisionTrees/'
PATH_OUTPUT = '/home/elyne/git/rdocChallenge/output/'
PATH_TEMP = '/home/elyne/git/rdocChallenge/temp/'

PATH_UNANNOTATED = '/home/elyne/git/rdocChallenge/input/unannotated/'
PATH_PREPROCESSED_UNANNOTATED = '/home/elyne/git/rdocChallenge/input/unannotated/preprocessed/'

##check if folders exist, otherwise make them
def checkMap(mapName):
    if not os.path.exists(mapName):
        os.makedirs(mapName)

def makeFolders():
    checkMap(PATH_RESOURCES)
    checkMap(PATH_TRAIN)
    checkMap(PATH_TEST)
    checkMap(PATH_PREPROCESSED_TRAIN)
    checkMap(PATH_PREPROCESSED_TEST)
    checkMap(PATH_DECISION_TREE)
    checkMap(PATH_OUTPUT)
    checkMap(PATH_TEMP)
