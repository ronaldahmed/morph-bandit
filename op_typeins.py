from utils import STOP
import pdb

class Operation:
  def __init__(self,_name,_pos,segment,_token,_primitive=False):
    self.name = _name
    self.form = _token
    self.pos = _pos
    # self.rev_pos = max(0,len(_token)-_pos-1) # pos from the end, in case of INS at the end, rev_pos could be 0, so we max it to 0
    self.primitive = _primitive
    # self.mask = 0
    self.segment = segment

    # indexing from 1 for form chars
    # pos 0 > e.g. ins at the beggining
    # pos n+1 > e.g. ins at the end

    # if _primitive and _name!=STOP:
    #   for i in range(_pos,_pos + len(segment)):
    #     self.mask |= (1 << i)


  def __repr__(self):
    return "(%s,%d,%s)" % (self.name,self.pos,self.segment)

  # def update_mask(self,mask2,seg2):
  #   # to be called only for contiguous ops of the same type
  #   self.mask |= mask2
  #   n_form = len(self.form)
  #   new_seg = ["_"] * n_form
  #   for i in range(n_form):
  #     if self.mask & (1 << i) > 0:
  #       new_seg[-i-1] = self.form[-i-1]

  #   for i in range(n_form):
  #     if self.mask & (1 << i) > 0:
  #       self.segment[-i-1] = self.form[-i-1]
  #   self.segment = "".join(self.segment).strip("_")


  def __hash__(self):
    return hash((self.name,self.segment))

  def __eq__(self,op2):
    return (self.name,self.segment) == (op2.name,op2.segment)

  def __ne__(self,op2):
    return not(self==op2)



#########################################################################


class TypeInstance:
  def __init__(self,lemma,form,actions=[]):
    self.lemma = lemma
    self.form = form
    self.ops = actions