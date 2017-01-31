import config as cfg
import utils
import re

from string import punctuation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

class QAPair(object):
    """
    Question-answer pair
    
    This element contains the indexes for begin and end of the question, the question itself,
    and the answer (split up in separate sentences), and the begin and endindex of the entire answer
    """
    def __init__(self):
        self.question = ''
        self.begQue = -1
        self.endQue = -1
        
        self.answers = []
        self.begAns = -1
        self.endAns = -1
    
    def __repr__(self):
        return self.__str__()
    def __str__(self):
        qst = "Q:" + self.question + "(" + str(self.begQue) + "," + str(self.endQue) + ")"
        ast = "A:" + ", ".join(self.answers) + "(" + str(self.begAns) + "," + str(self.endAns) + ")"
        return qst + "\n" + ast + "\n"

class TextBits():
    """
    TextBits contains all types of different words used to detect something (negations, categories, ..)
    
    """
    def __init__(self):
        
        #negative answers 
        self.denials = {"no", "never", "segment", "negative", "wnl", "none"}
        
        #@madhumita: uncertain denial from doctor's perspective
        self.conflicted = {"none reported", "denies", "denial", "none noted", "uncertain", "not sure"}
        
        #positive answers - filter to differentiate from short descriptive
        self.positive = {'yes'} #accept
        
        #numerical continuous
        self.continuous = {'audit c total score',
                           'axis v highest 12 month',
                           'axis v lowest 12 month',
                           'age'}

        #pre-specified
        #@madhumita: Added 'f' as female in gender
        #@madhumita: in marital status: divorced and widowed not present in data as I see it. Added it.
        self.categorical = {'gender': {'male': 1, 'female': 2, 'm': 1, 'f':2} #, 
#                             'social history marital status':{'single':1, 'married':2, 'committed relationship':3, 'divorced':4, 'widowed':5}
                            }
        self.family={"family history","family psych history"}
        self.strip_nonint = re.compile('[^\d]')


class Segment(object):

    def __init__(self):

        self.empty = 0
        self.questionsep = re.compile(r"\?|:|\s-\s")
        self.dangeroustails = [":","?","e.g.",'"high"','"up"',"(social anxiety)","(phobias)"]

        # Only use this for splitting, not counting.
        # self.sentencesep = re.compile(r"(?<![-0])\.(?![0-9-])|[a-z\s\"][A-Z]|[0-9]+[A-Z]|\)[A-Z]")
        self.sentencesep = re.compile(r"(?<!(\.\s|WN|Wn))[\.\"\sa-zA-Z0-9\(\)-][A-Z][a-rt-z]")
        self.cap_to_words = re.compile(r"[a-z0-9\)\(.]+[A-Z]")
        self.cap_to_words_2 = re.compile(r"[A-Z]{3,}[a-z]")

        self.garbage = re.compile("[{0}]".format(punctuation))
    
    def has_dangerous_tail(self, line):
        for dangerTail in self.dangeroustails:
            if line.endswith(dangerTail):
                return True
        return False

    def _line(self, line):
        """
        Process a single line using regular expressions.

        @param line: the line to process
        @type line: string
        @return: a question and answer tuple, where question is a string,
        and answer is a list with an answer, which is also a string.
        """

        numsep = len(self.questionsep.findall(line))

        if numsep == 1:
            q, a = self.questionsep.split(line)
            return q, [a.lstrip()]
        elif numsep > 1:
            line = self.questionsep.split(line)
            return ":".join(line[:-1]), [line[-1].lstrip()]
        else:
            return "", [line.lstrip()]       
        

    def _tail(self, line):
        if (self.has_dangerous_tail(line)):
                return line, []

        line = line.split(":")
        q, a = line[:-1], line[-1]

        return " ".join(q), [a]

    def segment_corpus(self, texts):
        """
        Segments a list of texts into dictionaries by calling segment

        :param texts:
        :return:
        """

        return map(self.segment, texts)

    def segment(self, text):
        """
        Segments one of the texts in the Rdoc challenge into questions and answers.

        @param text: The text to segment (in the edited format returned by Text object)
        @return: A dictionary of questions as keys, and lists of answers.
        """
        
        
        emptylines = 0
        question = [""]
        answer = [[]]
        #contains begin, split and endindex
        indexes = [[]]

        numm = 0
        
        normalCase = 0
        specialCase = 0

        for sentence in text.get_sentences(with_begin_end=True):
            #a sentence here is a line with [content, beginId, endId]           
            line = sentence[0]

            # Remove any whitespace from the end and beginning of the line.
            line = line.strip()

            if not line:
                emptylines += 1
                continue
            
            ##Exceptions to the common rule of colon separation are defined here
            if line.lower().endswith('chief complaint / hpi chief complaint (patients own words)'):
                question.append('chief complaint / hpi chief complaint (patients own words)')
                answer.append([])
                bi = sentence[1]
                si = sentence[2]
                ei = sentence[2]
                indexes.append([bi, si, ei])
                emptylines = 0
                continue
            if line.lower().startswith('history of present illness and precipitating events'):
                question.append('history of present illness and precipitating events')
                answer.append([])
                bi = sentence[1]
                si = sentence[2]
                ei = sentence[2]
                indexes.append([bi, si, ei])
                emptylines = 0
                continue
            if 'psychiatric diagnosis interview' in line.lower():
                question.append('psychiatric diagnosis interview')
                answer.append([])
                bi = sentence[1]
                si = sentence[2]
                ei = sentence[2]
                indexes.append([bi, si, ei])
                emptylines = 0
                continue
            if line.lower().startswith('problems'):
                question.append('problems')
                answer.append([line.lower().replace('problems','')])
                bi = sentence[1]
                si = sentence[1]+8
                ei = sentence[2]
                indexes.append([bi, si, ei])
                emptylines = 0
                continue
            
            

            numm += 1
            
            ##check if line consists of q: a, if so, we add a question and afterwards an answer
            numsep = len(self.questionsep.findall(line))
            if (numsep > 0) & (not self.has_dangerous_tail(line)):
                q, a = self._line(line)
                question.append(q)
                answer.append(a)
                bi = sentence[1]
                si = sentence[1] + len(q)
                ei = sentence[2]
                indexes.append([bi, si, ei])
                #print(sentence[0][0:len(q)],"|", sentence[0][len(q):sentence[2]-sentence[1]])
                normalCase += 1
                continue
            else:
                # One or more empty lines: special case.
                specialCase += 1
                q, a = self._tail(line)
                
                bi = sentence[1]
                si = sentence[1] + len(q)
                ei = sentence[2]
                indexes.append([bi, si, ei])
                question.append(q)
                answer.append(a)
                
                continue

            # Reset the line counter.
            emptylines = 0                

        results = []

        #Test printouts
        #print(normalCase, " normal cases, ", specialCase, "special cases.")
        #print(len(question), len(answer), len(indexes))


        # Treat the sentence after the bound separately.
        for q, a, ind in zip(question, answer, indexes):
            #print(q, a, ind)
            if not ind:
                continue

            q = " ".join([x for x in self.garbage.sub(' ', q).lower().split() if x not in ENGLISH_STOP_WORDS])
            a = [x.lower().strip() for x in a if x.strip()]

            # Treat "Chief Complaint", which contains the
            # patients own description as a separate question.
            
            if q.startswith("medications"):
                temp = q.split()
                q = temp[0]
                answerAppendix = " ".join(temp[1:])
                a.append(answerAppendix)
                ind[1] = ind[1] - len(answerAppendix) #the split point shifts to the left

            if a and a[0].endswith('Chief Complaint / HPI Chief Complaint ( Patients own words )'):
                item = QAPair()
                item.question = '(Chief Complaint / HPI Chief Complaint ( Patients own words )'
                item.answers = a[1:]
                item.begQue = ind[0]
                item.endQue = ind[1]
                item.begAns = ind[1]+1
                item.endAns = ind[2]
                results.append(item)
            elif not q and (len(results) > 0):
                #extended it so results needs to contain something already to call this function
                results[-1].answers.extend(a)
                results[-1].endAns = ind[2]
            elif not a and (len(results) > 0):
                ##only append question if previous answer is blank!
                if (not results[-1].answers):
                    results[-1].question += ' ' + q
                    results[-1].endAns = ind[2]
                    results[-1].endQue = ind[2]
                else:
                    item = QAPair()
                    item.question = q
                    item.answers = a
                    item.begQue = ind[0]
                    item.endQue = ind[1]
                    item.begAns = ind[1]+1
                    item.endAns = ind[2]
                    results.append(item)
            else:
                item = QAPair()
                item.question = q
                item.answers = a
                item.begQue = ind[0]
                item.endQue = ind[1]
                item.begAns = ind[1]+1
                item.endAns = ind[2]
                results.append(item)
        return results

if __name__ == '__main__':
    #TEST CODE
    data = utils.readData(cfg.PATH_TRAIN, cfg.PATH_PREPROCESSED_TRAIN)
    s = Segment()
    for d in data:
        text = d.getTextObject()
        segments = s.segment(text)
#         
#         print(segments)

#         for segment in segments:
#             
#             print("Question in TextObject:", text.get_covered_tokens(segment.begQue, segment.endQue))
#             print("Question concepts:", text.get_covered_concepts(segment.begQue, segment.endQue))
#             print("Question in segment:", segment.question)
#     
#             print("Answer in TextObject:", text.get_covered_tokens(segment.begAns, segment.endAns))
#             print("Answer concepts:", text.get_covered_concepts(segment.begAns, segment.endAns))
#             print("Answer in segment:", ", ".join(segment.answers))