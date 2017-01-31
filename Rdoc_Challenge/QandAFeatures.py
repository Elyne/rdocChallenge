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


Created on 8 Jul 2016

'''
import utils
from preprocessing.Segment import Segment, TextBits

import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import copy

from collections import defaultdict
from reach import Reach

from Resources import Resources
from UMLSFeatures import UMLSFeatures

class QandAFeatures(object):
    '''
    This class contains functions that generate features based on different variations,
    They are all based on question/answer pairs
    '''
    def __init__(self, expSet, trainSet, groupTrainQues = False, findSimTestQ=False):

        """
        Class for featurization based on Q-and-A

        :param featList: A list of features to apply.
        :param trainSet: A list of training that can be used to determine the questions and/or question clusters used
        """
        self.groupTrainQues = groupTrainQues
        self.retrieveSimilarQuestions = findSimTestQ

        self.segmenter = Segment()
        self.textBits = TextBits()

        self.expSet = expSet

        featuredict = {"BOW_ANSWERS": self.get_bow_answers,
                       "CATEGORICAL_QUESTIONSET": self.get_categorical_questionset,
                       "QUESTIONSET": self.get_questionset,
                       "PREAMBLE_CLUSTERS": self.get_preamble_clusters,
                       "CONCEPT_CLUSTERS":self.get_concept_clusters,
                       "LONG_QUESTIONSET":self.get_long_questionset,
                       "CONCEPTS_FROM_SUMMARY": self.get_concepts_from_summary}

        #questionSet - > yes/no/uncertain/else QSet
        #descQuesSet add
        self.featList = {f: featuredict[f.upper()] for f in expSet.featTypes if f.upper() in featuredict}

        # Determine features.
        # Gets questions from trainingset. {Questions:freq in train data}
        self.questionSet = dict()

        # make a list of segments found in the entire trainingset

        segments = []
        self.segConPairs = dict() #used for concept-based question clusterer, (a reconstructed list of concepts that occur in a question)
        for d in trainSet:
            currSegments = self.segmenter.segment(d.getTextObject())
            for currSegment in currSegments:
                currConInSegment = set()
                for concept in d.getTextObject().get_covered_concepts_annots(currSegment.begQue, currSegment.endQue):
                    nc = copy.copy(concept)
                    #removing prefixes if there are any
                    if ('_' in nc.ide):
                        nc.ide = nc.ide[nc.ide.index('_')+1:len(nc.ide)]
                    currConInSegment.add(nc)
                self.segConPairs[currSegment.question] = currConInSegment
                #self.segConPairs.append([currSegment.question, currConInSegment])
            segments.extend(currSegments)
        self.questionSet = Counter([seg.question.upper() for seg in segments])

        #for each q in qset assign categories - short/long etc
        #utils.out("Identifying question type")
        self.all_unique_questions, self.all_questions_type, self.all_questions_cat = self.identify_ques_type(trainSet, thresholdYNU=0.3,thresholdCat=0.3, thresholdLong=0.3)

        ## Now, doing the same for groups of questions occurring in train
        if self.groupTrainQues:
            utils.out("Grouping questions")
            self.grouped_questions, self.questions_type, self.grouped_questions_cat = self.get_grouped_questions(trainSet, simThreshold=0.9)
            utils.out("Assigning question type to grouped questions")
            self.grouped_questions_type = self.assign_ques_type(self.questions_type, 0.14, len(trainSet),thresholdYNU=0.3, thresholdCat=0.3, thresholdLong=0.3) #14%: smallest data set size

            utils.out("Done preprocessing questions, now clustering")

        ## Getting clusters based on the preamble
        if ('PREAMBLE_CLUSTERS' in self.featList):
            self.calcFalseulatePreambleClusters()

        ## Getting clusters based on the concepts in the questions
        if ('CONCEPT_CLUSTERS' in self.featList):
            self.calculateCommonConceptClusters()
        #utils.out("Done calculating clusters")

    def get_grouped_questions(self, trainSet, simThreshold):

        grouped_questions = defaultdict(list) #{id:[list of similar questions, where each item is a list of covered tokens in the question]}
        questions_type = defaultdict(lambda : defaultdict(int))
        grouped_questions_cat = defaultdict(set)

        for d in trainSet:
            cur_segment = self.segmenter.segment(d.getTextObject())
            for qap in cur_segment:
                qid = len(grouped_questions.keys())
                cur_q_tokens = d.getTextObject().get_covered_tokens(qap.begQue, qap.endQue)

                if any(cur_q_tokens in val for val in grouped_questions.values()):
                    continue
                qVec = Resources.getWordVectors().vectorize(cur_q_tokens, remove_oov=True)
                if not qVec:
                    continue
                norm_q_vec =  Reach.normalize(np.mean(qVec, axis=0))

                k = self.get_grouped_qid(norm_q_vec, grouped_questions, simThreshold)
                if k is not None:
                    qid = k

                grouped_questions[qid].append(cur_q_tokens)
                ansType, cat = self.get_ans_type(qap.answers)

                if not ansType:
                    continue

                questions_type[qid][ansType] += 1

                if cat:
                    grouped_questions_cat[qid].add(cat)

        return (grouped_questions, questions_type, grouped_questions_cat)

    def get_grouped_qid(self, norm_q_vec, grouped_questions, simThreshold):
        for k, q_tokens_list in grouped_questions.items():
            for t_list in q_tokens_list:
                if not Resources.getWordVectors().vectorize(t_list, remove_oov=True):
                    continue
                if np.dot(norm_q_vec, Reach.normalize(np.mean(Resources.getWordVectors().vectorize(t_list, remove_oov=True), axis=0))) >= simThreshold:
                    return k

        return None


#     def assign_grouped_q_type(self, questions_type, dataSet, thresholdYNU, thresholdCat, thresholdLong):
#         grouped_questions_type = defaultdict(lambda : defaultdict(int))
#
# #         for qid in questions_type:
#         self.assign_ques_type(questions_type, 0.14, len(dataSet),thresholdYNU, thresholdCat, thresholdLong) #14%: smallest data set size
#
#         return grouped_questions_type

    def get_ans_type(self, ans):
        '''
        Identify question type - where it is YNU, short desc, or long ans.
        Label 1 means negated, 2 means denied but uncertain, 3 means approved answer, 4 means single word answer, and 5 means long answer
        If short descriptive, return answer along with category label
        @return questionType, ansCategory
        '''
        if len(ans) == 0:
            return (0, None)

        conflicted = False

        for term in self.textBits.conflicted:
            if all(sub_term in ans[0].split() for sub_term in term.split()):
                    conflicted = True
                    break

            if conflicted:
                return (2, None)

            negated = bool(set(ans[0].split()).intersection(self.textBits.denials))
            if negated:
                return (1, None)

            approved = bool(set(ans[0].split()).intersection(self.textBits.positive))
            if approved:
                return (3, None)

        if len(ans) == 1 and len(ans[0].split()) == 1: #short desc
            return (4, ans[0].split()[0])
        else:
            return (5, None)

    def assign_ques_type(self, all_questions, minTh, lenData, thresholdYNU, thresholdCat, thresholdLong):
        all_questions_type = defaultdict(lambda : defaultdict(int))

        for ques in all_questions:
            if sum(all_questions[ques].values()) < minTh*lenData:
                continue

            type_cat = all_questions[ques][4]
            type_other = all_questions[ques][5]
            total_freq = sum(all_questions[ques].values())
            type_yes_no_uncertain = total_freq - type_cat - type_other

            #If fraction of qtype:1/2/3 > threshold: treat as yes/no/uncertain question.
            if type_yes_no_uncertain/total_freq >= thresholdYNU:
                all_questions_type[ques]['YNU'] = 1

            if type_cat/total_freq >= thresholdCat:
                all_questions_type[ques]['CAT'] = 1

            if type_other/total_freq >= thresholdLong:
                all_questions_type[ques]['LONG'] = 1

        return all_questions_type

    def identify_ques_type(self, dataSet, thresholdYNU, thresholdCat, thresholdLong):
        '''
        Identifies if a question in the dataSet should be processed as a yes/no/uncertainDenial (YNU) question,
        a categorical question, or a long answer question. These types are not mutually exclusive.
        YNU q-a pairs are cases with mention of yes/no/uncertainDenial terms (pre-compiled) are present in the first answer sentence.
        Categorical q-a-pair represents single word answers, where each answer is a separate category.
        Long Q-A pair are the cases where the answer is longer than one word.
        Moreover, the questions are chosen to be features of these types only if they occur more than 14% of the time in the data-set,
        and at least a certain percentage of the instances of this question in the data-set are of this type.
        @return all_questions_type: {ques:{type:i}} type:i represents process as the type "type"
        @return all_questions_cat {ques:{cat}} set of categories for a categorical question
        '''
        all_unique_questions = dict()
        all_questions = defaultdict(lambda : defaultdict(int))
        all_questions_cat = defaultdict(set)

        qId = 0
        for d in dataSet:
            cur_segment = self.segmenter.segment(d.getTextObject())
            for qap in cur_segment:
                qTokens = d.getTextObject().get_covered_tokens(qap.begQue, qap.endQue)
                ans = qap.answers

                if not any(qTokens in val for val in all_unique_questions.values()):
                    all_unique_questions[qId] = qTokens
                    qId += 1
#                     continue
#
#                 if qTokens not in all_unique_questions.values():


                ansType, cat = self.get_ans_type(ans)

                if ansType == 0:
                    continue
                #all_questions[qId][ansType] += 1
                #not using ids to identify questions, but the question itself!
                all_questions[qap.question][ansType] += 1

                if cat:
                    all_questions_cat[qap.question].add(cat)

        all_questions_type = self.assign_ques_type(all_questions, 0.14, len(dataSet),thresholdYNU, thresholdCat, thresholdLong) #14%: smallest data set size

        return (all_unique_questions,all_questions_type, all_questions_cat)



    def calculatePreambleClusters(self):
        oneWord = [((y[0], y[1])) for y in (Counter([' '.join(x.split(' ')[0:1]) for x in list(self.questionSet.keys()) if len(x.split(' ')) > 0])).most_common() if y[1] > 2]
        twoWord = [((y[0], y[1])) for y in (Counter([' '.join(x.split(' ')[0:2]) for x in list(self.questionSet.keys()) if len(x.split(' ')) > 1])).most_common() if y[1] > 2]
        threeWord = [((y[0], y[1])) for y in (Counter([' '.join(x.split(' ')[0:3]) for x in list(self.questionSet.keys()) if len(x.split(' ')) > 2])).most_common() if y[1] > 2]

        self.preambleClusters = []
        for pair1 in oneWord:
            if (pair1[1] > 2):
                p2 = False
                for pair2 in twoWord:
                    if pair2[0].startswith(pair1[0]):
                        #we check if the frequency of pair2 is equal to that of pair3, if so, we add pair 3 as a cluster
                        p3 = False
                        for pair3 in threeWord:
                            if pair3[0].startswith(pair2[0]):
                                if pair2[1] == pair3[1]:
                                    p3 = True
                                    self.preambleClusters.append(pair3)
                                    if (pair1[1] == pair2[1]):
                                        p2 = True
                        if not p3:
                            if pair1[1] == pair2[1]:
                                #we check if the frequency of pair2 is equal to pair1, is so, pair 2 is a cluster
                                p2 = True
                                self.preambleClusters.append(pair2)
                if (not p2) & (pair1[1] > 3):
                    #else pair 1 is a cluster (if over 3 questions have it)
                    self.preambleClusters.append(pair1)

    '''
    This function is commented out, as fp-growth is removed from the project
    def calculateCommonConceptClusters(self):
        ##First, we make a list of all transactions
        transactions = []
        for que in self.segConPairs:
            transaction = self.segConPairs[que]
            ##Cheaty bit: we make a transaction have unique items by checking if its description is already in the item.
            transaction2 = list(set([str(x) for x in transaction]))
            #print(transaction, "|", transaction2)
            if bool(transaction):
                transactions.append(transaction2)

        result = []
        for itemset, support in find_closed_itemsets(transactions, 5):
            result.append((itemset,support))
        self.conceptClusters = sorted(result, key=lambda i: i[0])
        print(self.conceptClusters)'''

    def apply(self, document):
        """
        This function wraps getting all kinds of QandA-based features. A
        segmenter is first applied here, then the selected features are calculated and added to the sample.

        Applies all functions specified in the constructor (self.features)
        to the incoming documents and stores the generated features.
        Each featurevalue has the name of that feature appended to it, to avoid
        collisions (e.g. a bow vectorizer might contain a word that is also a question)

        :param document: the Data object you want to add the features to
        :return: A list of dictionaries, representing the featurized counterparts of
            the documents
        """

        segments = self.segmenter.segment(document.getTextObject())

        ##Now deriving the different types of features

        # List of featurized documents
        featurized = dict()

        # Feature list for single document.

        # Iterate over pre-defined functions.
        for name, function in self.featList.items():
            #utils.out(name)
            # Apply function and change name based on function
            # to avoid collisions.
            if name == "BOW_ANSWERS":
                featurized[name] = {"_".join([name, k]): v for k, v in function(segments, document).items()}
            elif name == "CATEGORICAL_QUESTIONSET":
                featurized[name] = {"_".join([name, k]): v for k, v in function(segments, document).items()}
            elif name == "QUESTIONSET":
                featurized[name] = {"_".join([name, k]): v for k, v in function(segments, document).items()}
            elif name == "LONG_QUESTIONSET":
                featurized[name] = {"_".join([name, k]): v for k, v in function(segments, document).items()}
            elif name == "CONCEPTS_FROM_SUMMARY":
                featurized[name] = {"_".join([name, k]): v for k, v in function(segments, document).items()}
            else:
                featurized[name] = {"_".join([name, k]): v for k, v in function(segments).items()}

        # return the bin of featurized segments
        return featurized


    def get_bow_answers(self, segment, doc):
        """
        Bag of words representation of answers.

        :param segment: a list of QApairs
        :return: A list of word counts.
        """
        answers = []
        for y in segment:
            for answer in y.answers:
                answers.extend(answer.split())

        return Counter([x for x in answers if x not in ENGLISH_STOP_WORDS])

    def get_categorical_questionset(self, segment, doc, usingPreEval = True):
        """
        Featurizes the answers.

        Featurizes:

            'audit c total score'
            'axis v highest 12 month'
            'axis v lowest 12 month'
             'axis v gaf current' @madhumita: not treated as continuous!
            'age'
            'cups day'

            separately as continuous variables.

        Featurizes:

            'gender'
            'social history marital status'
            @madhumita: treat as short descriptive instead 'axis ii personality disorders'

            separately as categorical variables

        For each question, this function returns a 3 if the answer does not
        contain a denial or is otherwise normal (WNL) or is longer than 1 sentence.
        If the answer is a certain negation, it returns a 1.
        If the answer is an uncertain negation, it returns a 2.
        0 is reserved for unknown questions.

        :param segment: a dictionary, representing the questions and answers
            from a single document.
        :return: A dictionary of integer values.
            For binary values: 1 indicates a no (certain),
                2 indicates a no (uncertain)
                3 indicated a yes
                and a 4 indicates something else.
            For categorical values: categories are defined above.
            For continuous variables: the value as an integer.
        """

        featurized = {}

        for qap in segment:
            k = qap.question
            v = qap.answers

            if len(v) == 0:
                continue

#         print(vec)
            # Continuous variables on an integer scale.
            if k in self.textBits.continuous:

                try_int = self.textBits.strip_nonint.sub(" ", v[0]).split()
                if not try_int:
                    utils.out("q:{0} a:{1} no good for continuous variable".format(k, try_int))
                    # No integers in answer
                    continue
                featurized[k] = int(try_int[0])
                continue

            if usingPreEval:
                if self.all_questions_type[k]['CAT'] == 1:
                    if v and v[0] in self.all_questions_cat[k]:
                        featurized[k] = v[0]
                        continue
                    else:
                        featurized[k]=0
            else:
                # Categorical variables with predefined values.
                if k in self.textBits.categorical:

                    try:
                        featurized[k] = self.textBits.categorical[k][v[0]]
                    except (KeyError, TypeError):
                        utils.out("q:{0} a:{1} no good for categorical variable".format(k, v))
                        featurized[k] = 0
                    continue


            # False if there is a single answer and
            # this answer contains a denial.
            #binarized = bool(len(v) == 1 and set(v[0].split()).intersection(self.textBits.denials))

            # Add one to the integer representation
            # so that False == 1 and True == 2.
            # 0 is reserved for Unknown.
            #featurized[k] = str(binarized)

        return featurized

    def get_questionset(self, segment, doc, usingPreEval = True):
        """
        Answers a set of regularly occurring answers with a 0 ('unknown', not present), a 1 if it is a negative answer,
        2 if it is negative but uncertain, and a 3 if it is known, and is not negative/uncertain.

        For each question, this function returns a 3 if the answer does not
        contain a denial or is otherwise normal (WNL) or is longer than 1 sentence.
        If it does not satisfy these  conditions, it returns a 1 or 2 for the
        question, depending on whether it is an uncertain negation or a certain negation.
        0 is reserved for unknown questions.

        :param segment: a list of QAPair objects, representing the questions and answers
            from a single document.
        :return: A dictionary of integer values.
            For binary values: 1 indicates a negative answer,
                2 indicated negative answers, but maybe uncertain,
                3 indicated approval or positive answers,
                and a 4 indicates something else.
            For categorical values: categories are defined above.
            For continuous variables: the value as an integer.
        """

        featurized = {}

        if self.retrieveSimilarQuestions and self.groupTrainQues:
            qSet = self.grouped_questions_type
            questions = self.grouped_questions
        else:
            qSet = self.all_questions_type
            questions = self.all_unique_questions

        for qap in segment:
            if self.retrieveSimilarQuestions:
                k = self.get_group_id(questions, doc.getTextObject().get_covered_tokens(qap.begQue, qap.endQue), 0.9)
                if k is None:
                    continue
                else:
                    k = str(k)
            else:
                k = qap.question

            v = qap.answers

            if len(v) == 0:
                continue

            if usingPreEval and not qSet[k]['YNU'] == 1:
                    continue

            for term in self.textBits.conflicted:
                if all(sub_term in v[0].split() for sub_term in term.split()):
                    conflicted = True
                    break
            conflicted = False

            if conflicted:
                featurized[k] = 2
                continue

            negated = bool(set(v[0].split()).intersection(self.textBits.denials))
            if negated:
                featurized[k] = 1
                continue

            approved = bool(set(v[0].split()).intersection(self.textBits.positive))
            if approved:
                featurized[k] = 3
                continue

        if k is not None:
            featurized[k] = 4
            #featurized.update(Counter(["{0}-{1}".format(k, s) for s in " ".join(v).lower().split()]))

        return featurized

    def getConceptsForRange(self, doc, begin, end, filt = 'DSM+1'):
        '''
        This function gets all concepts between begin and end, and then applies the named filter to it
        '''
        uf = UMLSFeatures()

        unique = []
        for conc in doc.getTextObject().get_covered_concepts(begin,end):
            if (filt == 'DSM+1'):
                ans = uf.isRemotelyPsychiatric(conc)
                if ans[0]:
                    unique.append(ans[1])
            #TODO: add other filters as well
        return unique

    def get_long_questionset(self, segment, doc, usingPreEval=True):
        """
        Answers a set of regularly occurring answers with the concepts that are defined in it

        :param segment: a list of QAPair objects, representing the questions and answers
            from a single document.
        :return: A dictionary of concepts, that contain all concepts from questions that are considered 'long'

        """

        featurized = {}

        ##This does not use the long question predefinition system, but rather considers every answer with more than 3 tokens
        # it does not use long question predefinition because the system is insufficient: it only returns 'long' when a question is NOT YNU or descriptive
        # this should preferably be changed in the future

        for qap in segment:
            k = qap.question
            #v = qap.answers

            if (len(qap.answers) > 0):
                if usingPreEval:
                    if self.all_questions_type[k]['LONG'] == 1:
                        for conc in self.getConceptsForRange(doc, qap.begQue,qap.endAns):
                            try:
                                featurized[conc] += 1
                            except:
                                featurized[conc] = 1
                elif (len(qap.answers) > 1) | (len(qap.answers[0].split(' ')) > 3):
                            for conc in self.getConceptsForRange(doc, qap.begQue,qap.endAns):
                                try:
                                    featurized[conc] += 1
                                except:
                                    featurized[conc] = 1

        return featurized

    def get_concepts_from_summary(self, segment, doc):
        """
        Gives a list of concepts from the DSM+1 subset that occur in the questions indicating a summary set.

        The specific summary set is also included within the feature name, so different summaries is working.

        """
        featurized = {}

        for qap in segment:
            if ('chief complaint hpi chief complaint patients words' in qap.question.lower()):
                for conc in self.getConceptsForRange(doc, qap.begAns,qap.endAns):
                    try:
                        featurized["chiefComplaint-" + conc] += 1
                    except:
                        featurized["chiefComplaint-" + conc] = 1
            elif ('formulation' in qap.question.lower()):
                for conc in doc.getTextObject().get_covered_concepts(qap.begAns,qap.endAns):
                    try:
                        featurized["formulation-" + conc] += 1
                    except:
                        featurized["formulation-" + conc] = 1
            elif ('history present illness precipitating events' in qap.question.lower()):
                for conc in doc.getTextObject().get_covered_concepts(qap.begAns,qap.endAns):
                    try:
                        featurized["histPresIll-" + conc] += 1
                    except:
                        featurized["histPresIll-" + conc] = 1
            elif ('problems' in qap.question.lower()):
                for conc in doc.getTextObject().get_covered_concepts(qap.begAns,qap.endAns):
                    try:
                        featurized["prob-" + conc] += 1
                    except:
                        featurized["prob-" + conc] = 1
            elif ('medical history' in qap.question):
                for conc in doc.getTextObject().get_covered_concepts(qap.begAns,qap.endAns):
                    try:
                        featurized["medHist-" + conc] += 1
                    except:
                        featurized["medHist-" + conc] = 1
            elif ('multi axial diagnoses assessment axis code description' in qap.question.lower()):
                for conc in doc.getTextObject().get_covered_concepts(qap.begAns,qap.endAns):
                    try:
                        featurized["diagnoses-" + conc] += 1
                    except:
                        featurized["diagnoses-" + conc] = 1
        return featurized


    def get_group_id(self, questions, qTokens, simThreshold=0.9):

        vec = Resources.getWordVectors().vectorize(qTokens, remove_oov=True)
        if not vec:
            return None

        qVec = Reach.normalize(np.mean(vec, axis = 0))
        mostSimQ = None
        maxSim = 0.0

        for groupId, groupQTokens in questions.items():
            for cur_q_tokens in groupQTokens:
                cur_vec = self.expSet.getWordVectors().vectorize(cur_q_tokens,remove_oov=True)
                if not cur_vec:
                    continue
                curSim = np.dot(qVec, Reach.normalize(np.mean(cur_vec, axis = 0)))
                if curSim > maxSim:
                    maxSim = curSim
                    mostSimQ = groupId

        if maxSim >= simThreshold:
            return mostSimQ
        else:
            return None



    def get_cluster_score(self, answers, featurized, clusterName):
        """
        Common function for all cluster types to give a score on the answers that a cluster has
        Only call the function if the question is considered part of the cluster!

        """
        if (len(answers) > 0):
            if bool(set(answers[0].lower().split()).intersection(self.textBits.denials)):
                None
            elif bool(set(answers[0].lower().split()).intersection(self.textBits.conflicted)):
                featurized[clusterName] = featurized[clusterName] + 0.5
            else:
                featurized[clusterName] = featurized[clusterName] + 1
        return featurized


    def get_concept_clusters(self, doc):
        """
        This will look if a question occurs in a preambled cluster, and score the clusters for the entire document on that.

        For now, if the question occurs, and is not a denial, it will count as +1, otherwise there will be no change to the local score.
        We normalise by the expected number of questions.

        """
        featurized = dict()
        for cluster in self.conceptClusters:
            featurized['_'.join(cluster[0])] = 0
        for qap in doc:
            try:
                concepts = set([str(x) for x in self.segConPairs[qap.question]])

                #go over the clusters, check if concept representations are all part of it
                allIn = True
                for cluster in self.conceptClusters:
                    for concept in concepts:
                        if not (concept in cluster):
                            allIn = False
                            break
                    if allIn == True:
                        featurized = self.get_cluster_score(qap.answers, featurized, '_'.join(cluster[0]))
            except:
                continue
        return featurized

    def get_preamble_clusters(self, doc):
        """
        This will look if a question occurs in a preambled cluster, and score the clusters for the entire document on that.

        For now, if the question occurs, and is not a denial, it will count as +1, otherwise there will be no change to the local score.
        We normalise by the expected number of questions.

        """
        featurized = dict()
        for cluster in self.preambleClusters:
            featurized[cluster[0]] = 0
        
        for qap in doc:
            for cluster in self.preambleClusters:
                if (qap.question.upper().startswith(cluster[0])):
                    featurized = self.get_cluster_score(qap.answers, featurized, cluster[0])

        #normalise
        for cluster in self.preambleClusters:
            featurized[cluster[0]] = featurized[cluster[0]] / cluster[1]
        return featurized
