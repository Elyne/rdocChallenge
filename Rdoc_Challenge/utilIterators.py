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

Created on 15 Jun 2016

'''
import config as cfg

'''
This class creates an iterative sliding window over a range of tokens
'''
class WindowSlider():
    def __init__(self, tokens, windowSize=5):
        self.tokens = [x for x in tokens if x not in {'',' ', '\s','<newline>',',','.','<tab>',':','\n',';','?','!','-','\\','"',"'",')','_','*'}]
        self.windowSize = windowSize
        
    def __iter__(self):
        for i in range(0,len(self.tokens)):
            window = []
            for token in self.tokens[i:]:
                #ignore separator tokens
                window.append(token)
                if len(window) == self.windowSize:
                    break

            while len(window) < self.windowSize:
                window.append('_')
            yield window
                
'''Iterating over a table in a database. Keep in mind that the id should be the first index, and the description/ string the second
Example: [CUI, str]
'''
class UMLSTableIterator():
    def __init__(self, query):
        self.q = query
    def __iter__(self):
        conn = cfg.connUMLS()
        cursor = conn.cursor()
        cursor.execute(self.q)
        row = cursor.fetchone()
        while row:
            yield [row[0], row[1]]
            row = cursor.fetchone()
        conn.close()
    
class FileIterator():
    def __init__(self, fName, idSep=";", tokenSep=",", splitTokens=True):
        self.fName = fName
        self.idSep = idSep
        self.tokenSep = tokenSep
        self.splitTokens = splitTokens
    def __iter__(self):
        for line in open(self.fName,'r'):
            line = line.replace("\n","")
            items = line.split(self.idSep)
            if (len(items) > 1):
                if (self.splitTokens):
                    yield [items[0], items[1].split(self.tokenSep)]
                else:
                    yield items