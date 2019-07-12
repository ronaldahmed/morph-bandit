import sys
import warnings

UNK_TOKEN = "*UNK*"

class LabelDictionary(dict):
    '''This class implements a dictionary of labels. Labels as mapped to 
    integers, and it is efficient to retrieve the label name from its 
    integer representation, and vice-versa.'''
    def __init__(self, label_names=[]):
        self.names = []
        self._add = True
        for name in label_names:
            self.add(name)

    def add(self, name):
        label_id = len(self.names)
        if name in self:
            # warnings.warn('Ignoring duplicated label ' +  name) 
            return self[name]
        elif not self._add:
            return self[UNK_TOKEN]
        self[name] = label_id
        self.names.append(name)
        return label_id

    def get_label_name(self, label_id):
        return self.names[label_id]

    def get_label_id(self, name):
        return self.get(name,self[UNK_TOKEN])
