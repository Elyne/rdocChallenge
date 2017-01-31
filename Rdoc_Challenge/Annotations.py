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


'''
class Token(object):

    def __init__(self, begin, end, string, pos):
        self.begin = begin
        self.end = end
        self.str = string
        self.pos = pos

    def __repr__(self):

        return self.str


class Span(object):

    def __init__(self, begin, end, covered_tokens):

        self.begin = begin
        self.end = end
        self.covered = covered_tokens

    def __repr__(self):
        '''
        Cannot just be ' '.join(), because the spacing between elements is not always equal!
        '''
        stri = ''
        if (len(self.covered) > 0):
            ind = self.covered[0].begin
            for token in self.covered:
                while ind < token.begin:
                    stri += ' '
                    ind += 1
                stri += token.str
                ind = token.end
        return stri


class Sentence(Span):
    def __init__(self, begin, end, covered_tokens):
        super(Sentence, self).__init__(begin=begin, end=end, covered_tokens=covered_tokens)


class Chunk(Span):
    
    def __init__(self, begin, end, covered_tokens, chunk_type):
        super(Chunk, self).__init__(begin=begin, end=end, covered_tokens=covered_tokens)
        self.chunkType = chunk_type
        

class Concept(Span):
    
    def __init__(self, begin, end, ide, certainty, words):

        super(Concept, self).__init__(begin=begin, end=end, covered_tokens=words)
        self.ide = ide
        self.certainty = certainty