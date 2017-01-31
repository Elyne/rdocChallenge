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

from preprocessing.Segment import Segment, TextBits
import copy

class Text(object):

    def __init__(self, content, chunks, tokens, sentences, concepts):
        self.content = content
        self.chunks = chunks
        self.tokens = tokens
        self.sentences = sentences
        self.concepts = concepts

    def get_sentences(self, with_begin_end=False):
        nsentences = []

        for sentence in self.sentences:
            c = [sentence.__repr__()]
            if with_begin_end:
                c.append(sentence.begin)
                c.append(sentence.end)
            nsentences.append(c)
        return nsentences

    def get_tokens(self):

        return [t.str for t in self.tokens]

    def get_covered_tokens(self, begin, end):
        tokens = []
        for token in self.tokens:
            if token.begin >= begin and token.end <= end:
                tokens.append(token.str)
        return tokens

    def get_pos_tokens(self):

        return [(t.str, t.pos) for t in self.tokens]

    def get_covered_pos_tokens(self, begin, end):

        pos_tokens = []
        for t in self.tokens:
            if t.begin >= begin and t.end <= end:
                pos_tokens.append((t.str, t.pos))

        return pos_tokens

    def get_chunks(self, pos=True, with_begin_end=False):
        chunks = []

        for chunk in self.chunks:
            if pos:
                c = [[(x.str, x.pos) for x in chunk.covered], chunk.chunkType]
            else:
                c = [[x.str for x in chunk.covered], chunk.chunkType]
            if with_begin_end:
                c.append(chunk.begin)
                c.append(chunk.end)
            chunks.append(c)

        return chunks

    def get_concepts(self):

        return [c.ide for c in self.concepts]
    
    def get_covered_concepts(self, begin, end):

        return [c.ide for c in self.get_covered_concepts_annots(begin, end)]
    
    def get_covered_concepts_annots(self, begin, end):
        concepts = []

        for concept in self.concepts:

            c_begin = concept.begin
            c_end = concept.end

            if begin <= c_begin and end >= c_end:
                concepts.append(concept)
        return concepts

    def get_non_denied_text(self):

        words = []
        segmenter = Segment()
        tb = TextBits()

        for segment in segmenter.segment(self):
            q_words = self.get_covered_tokens(segment.begQue, segment.endQue)
            a_words = self.get_covered_tokens(segment.begAns, segment.endAns)
            if segment.answers:
                if not set(segment.answers[0].lower().split()).intersection(tb.denials):
                    words.extend(a_words)
                else:
                    words.extend(q_words)

        return words 
    
    
    
    def processConceptsOnContext(self, cueCollection, modifyInsteadOfRemove=False, prefix='DEN_', includeAnswers=True):
        '''
        Core function that tries a list of cues on all segments in the text. You can use it to detect uncertainty or denial, and you can either
        not add questions for which the answer is indicated as a cue or you can modify them by adding a prefix
        '''
        retainedConcepts = []
        segmenter = Segment()

        for segment in segmenter.segment(self):
            qCon = self.get_covered_concepts_annots(segment.begQue, segment.endQue)
            aCon = self.get_covered_concepts_annots(segment.begAns, segment.endAns)
            #print('qcon:',[con.ide for con in qCon])
            #print('acon:',[con.ide for con in aCon])
            if bool(segment.answers):
                # if not set(segment.answers[0].lower().split()).intersection(cueCollection):
                
                '''cueFound = False
                for cue in cueCollection:
                    if cue in segment.answers[0].lower():
                        cueFound = True
                        break
                    
                if not cueFound:'''
                if not set(segment.answers[0].lower().split()).intersection(cueCollection):
                    retainedConcepts.extend(qCon)
                    if (includeAnswers):
                        retainedConcepts.extend(aCon)
                else:
                    #do nothing if remove, add it with a prefix if modify
                    if modifyInsteadOfRemove:
                        for con in qCon:
                            nc = copy.copy(con)
                            nc.ide = prefix + con.ide
                            retainedConcepts.append(nc) 
                        if (includeAnswers):
                            for con in aCon:
                                nc = copy.copy(con)
                                nc.ide = prefix + con.ide
                                retainedConcepts.append(nc) 
                if not includeAnswers:
                    retainedConcepts.extend(aCon)
        return retainedConcepts
    
    def remove_concepts_from_denied_questions(self):
        """
        Function to remove concepts that occur in questions that are denied by the 
        It scans the question and answer for concepts, if the answer is "yes", or complicated, we keep the concepts in the question
        If the answer is 'No' or 'Wnl' (within normal limits), we ignore concepts detected in the question (if the answer is long, we keep the answer)
        """
        tb = TextBits()
        self.concepts = self.processConceptsOnContext(tb.denials)
        
    def separate_concepts_from_denied_questions(self):
        """
        Function to modify concept-ids that occur in questions that are denied by the 
        It scans the question and answer for concepts, if the answer is "yes", or complicated, we keep the concepts in the question as they are
        If the answer is 'No' or 'Wnl' (within normal limits), we add a prefix 'NEG_' to its concept id
        """
        tb = TextBits()
        self.concepts = self.processConceptsOnContext(tb.denials, modifyInsteadOfRemove=True, prefix='DEN_')
        
    def remove_concepts_from_uncertain_questions(self):
        """
        Function to remove concepts that occur in questions that are denied by the 
        It scans the question and answer for concepts, if the answer is "yes", or complicated, we keep the concepts in the question
        If the answer is uncertain, we ignore concepts detected in the question (if the answer is long, we keep the answer)
        """
        tb = TextBits()
        self.concepts = self.processConceptsOnContext(tb.conflicted)
        
    def separate_concepts_from_uncertain_questions(self):
        """
        Function to modify concept-ids that occur in questions that are denied by the 
        It scans the question and answer for concepts, if the answer is "yes", or complicated, we keep the concepts in the question as they are
        If the answer is uncertain, we add a prefix 'NEG_' to its concept id
        """
        tb = TextBits()
        self.concepts = self.processConceptsOnContext(tb.conflicted, modifyInsteadOfRemove=True, prefix='UNC_')
        
    def remove_concepts_from_family_questions(self):
        """
        Function to remove concepts that occur in questions that are denied by the 
        It scans the question and answer for concepts, if the answer is "yes", or complicated, we keep the concepts in the question
        If the answer is uncertain, we ignore concepts detected in the question (if the answer is long, we keep the answer)
        """
        tb = TextBits()
        retainedConcepts = []
        segmenter = Segment()

        for segment in segmenter.segment(self):
            qCon = self.get_covered_concepts_annots(segment.begQue, segment.endQue)
            aCon = self.get_covered_concepts_annots(segment.begAns, segment.endAns)
            #print('qcon:',[con.ide for con in qCon])
            #print('acon:',[con.ide for con in aCon])
            if bool(segment.answers):
                cueFound = False
                for cue in tb.family:
                    if cue in segment.question.lower():
                        cueFound = True
                        break
                    
                if not cueFound:
                    retainedConcepts.extend(qCon)
                    retainedConcepts.extend(aCon)
                #do nothing if remove, add it with a prefix if modify
        self.concepts = retainedConcepts
        
    def separate_concepts_from_family_questions(self):
        """
        Function to modify concept-ids that occur in questions that are denied by the 
        It scans the question and answer for concepts, if the answer is "yes", or complicated, we keep the concepts in the question as they are
        If the answer is uncertain, we add a prefix 'NEG_' to its concept id
        """
        tb = TextBits()
        prefix='FAM_'
        retainedConcepts = []
        segmenter = Segment()

        for segment in segmenter.segment(self):
            qCon = self.get_covered_concepts_annots(segment.begQue, segment.endQue)
            aCon = self.get_covered_concepts_annots(segment.begAns, segment.endAns)
            #print('qcon:',[con.ide for con in qCon])
            #print('acon:',[con.ide for con in aCon])
            if bool(segment.answers):
                cueFound = False
                for cue in tb.family:
                    if cue in segment.question.lower():
                        cueFound = True
                        break
                    
                if not cueFound:
                    retainedConcepts.extend(qCon)
                    retainedConcepts.extend(aCon)
                else:
                    #do nothing if remove, add it with a prefix if modify
                    for con in qCon:
                        nc = copy.copy(con)
                        nc.ide = prefix + con.ide
                        retainedConcepts.append(nc) 
                        for con in aCon:
                            nc = copy.copy(con)
                            nc.ide = prefix + con.ide
                            retainedConcepts.append(nc) 

        self.concepts = retainedConcepts
