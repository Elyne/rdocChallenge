# Treecounter
Our approach to the CEGS N-GRID 2016 Shared Task in Clinical Natural Language Processing, about the detection of symptom severity.

Written in Python 3.5

Authors: 
- Elyne Scheurwegs
- Madhumita Sushil
- Stéphan Tulkens
- Walter Daelemans
- Kim Luyckx

## Setting up the project
- Fix dependencies. We import a lot of stuff, but it is all installable through pip3
- First, change all paths in config.py to link to the data you want it to use, and lead everything to the correct path.
- You need to preprocess all .xml files from the challenge with cTakes first, and add them to a separate 'preprocessed' folder.
- Then, you need to run DictionaryMatcher on that folder. This will detect concepts with our custom algorithm. Be sure that you have a version of an UMLS lexicon in the appropriate file in your resources folder (examples included).
- Then, you are ready to run an evaluation script on your data. All configurable options are grouped in 'ExperimentSettings'.

## Evaluation scripts
Several modules are set up for using it in a testing setup.
- tenFoldEval: does a 10-fold cross-validation over training data and returns the metrics reached during cross-validation
- testSetup: performs the model for the training set. Creates the files with annotations as output, in a folder named after the set of features used.
- testEval: compares the output from testSetup (the actual files) to the gold standard.

## Generating UMLS subsets:
You need a working UMLS metathesaurus in a database to create UMLS subsets. The script in crawlers/generateExpertCorpora.py can be used to create the various subsets. They will end up in your configured resources folder. Examples of these corpora are already present.
