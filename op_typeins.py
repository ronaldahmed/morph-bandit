from utils import STOP

class Operation:
  def __init__(self,_name,_pos,_token,_primitive=False):
    self.name = _name
    self.pos = _pos
    self.form = _token
    self.rev_pos = max(0,len(_token)-_pos-1) # pos from the end, in case of INS at the end, rev_pos could be 0, so we max it to 0
    self.primitive = _primitive
    self.mask = 0
    self.segment = ''

    if _primitive and _name!=STOP:
      self.update_mask(1 << self.rev_pos)


  def __repr__(self):
    return "(%s,%d,%s,%d)" % (self.name,self.pos,self.segment,self.mask)

  def update_mask(self,mask2):
    self.mask |= mask2
    n_form = len(self.form)
    self.segment = ["_"] * n_form
    for i in range(n_form):
      if self.mask & (1 << i) > 0:
        self.segment[-i-1] = self.form[-i-1]
    self.segment = "".join(self.segment).strip("_")


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