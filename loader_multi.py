from data_utils import DataLoaderAnalizer
from utils import oplabel_pat, map_ud_folders \
                  to_cuda, fixed_var, \
                  PAD_ID, UNK_TOKEN, PAD_TOKEN, EOS, SOS
import json

TREEBANK_TEMPLATE = "2019/task2/%s/%s-um-%s.conllu"
ACTION_FILE_TEMPLATE = "data/%s/%s"
MULTI_LING_CONF_JSON = "multi_ling_conf.json"

def load_set_conf(tbset):
  conf = json.load(open(MULTI_LING_CONF_JSON,'r'))
  assert tbset in conf
  return conf[tbset]


class DataLoaderMLing(DataLoaderAnalizer):
  def __init__(self,args):
    super(DataLoaderMLing,self).__init__(args)
    special_tokens = [PAD_TOKEN,UNK_TOKEN]
    self.vocab_oplabel = LabelDictionary(label_names=special_tokens)
    self.vocab_feat_bundles = LabelDictionary(label_names=special_tokens)
    self.vocab_feats = LabelDictionary(label_names=special_tokens+[SOS,EOS])
    self.src_tbs = load_set_conf(args.src_set)
    self.tgt_tb = args.tgt


  def vocabs_input_mode_toggle(self,fill_vocab=False):
    self.vocab_oplabel._add = fill_vocab
    self.vocab_feat_bundles._add = fill_vocab
    self.vocab_feats._add = fill_vocab


  def read_action_file(self,tbname, split,fill_vocab=False):  
    self.vocabs_input_mode_toggle(fill_vocab)

    lang = tbname[:2]
    filename = ACTION_FILE_TEMPLATE % (tbname,split)
    sents,labels,lemmas,forms = [],[],[],[]
    sent,label,lem_sent,form_sent = [],[],[],[]
    for line in open(filename,'r'):
      line = line.strip('\n')
      if line=='':
        sents.append(sent)
        labels.append(label)
        forms.append(form_sent)
        lemmas.append(lem_sent)
        sent = []
        label = []
        lem_sent = []
        form_sent = []
        continue
      # w,lem,feats,ops = line.split("\t")
      # op_seq = ops.split(" ")
      comps = line.split("\t")
      w,lem,feats = comps[:3]
      op_ids = [self.vocab_oplabel.add(op) for op in comps[3:]]
      
      sent.append(op_ids)
      # lem_sent.append(self.vocab_lemmas.get_label_id(lem))
      # form_sent.append(self.vocab_forms.get_label_id(w))
      lem_sent.append(lem)
      form_sent.append(w)
      if self.args.tagger_mode=="bundle":
        label.append(self.vocab_feats.get_label_id(feats))
      else:
        sos = self.vocab_feats.get_label_id(SOS)
        eos = self.vocab_feats.get_label_id(EOS)
        label.append([sos] + [self.vocab_feats.get_label_id(feat) for feat in feats.split(";")] + [eos])
      #

    self.vocabs_input_mode_toggle(False)


  def load_data(self,fill_vocab=False):

    
    
    