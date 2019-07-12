from data_utils import DataLoaderAnalizer
from utils import oplabel_pat, uploadObject \
                  to_cuda, fixed_var, \
                  PAD_ID, UNK_TOKEN, PAD_TOKEN, EOS, SOS
import json

TREEBANK_TEMPLATE = "2019/task2/%s/%s-um-%s.conllu"
ACTION_FILE_TEMPLATE = "data/%s/%s"
TOKENIZED_JSON_TEMPLATE = "data/%s/%s-%s.json"  # tbset, tb, split
MULTI_LING_CONF_JSON = "multi_ling_conf.json"


def load_set_conf(tbset):
  conf = json.load(open(MULTI_LING_CONF_JSON,'r'))
  assert tbset in conf
  return conf[tbset]

def loader_from_pickle(args):
  filename = "data/%s-%s/loader.pickle" % (args.src_set,args.tgt)
  if os.path.exists(filename):
    return uploadObject(filename)
  else:
    return DataLoaderMLing(args)


class DataLoaderMLing(object):
  def __init__(self,args):
    special_tokens = [PAD_TOKEN,UNK_TOKEN]
    self.vocab_oplabel = LabelDictionary(label_names=special_tokens)
    self.vocab_feat_bundles = LabelDictionary(label_names=special_tokens)
    self.vocab_feats = LabelDictionary(label_names=special_tokens+[SOS,EOS])
    self.vocab_lemmas = LabelDictionary()
    self.vocab_forms = LabelDictionary()
    self.src_tbs = load_set_conf(args.src_set)
    self.tgt_tb = args.tgt

  def get_vocab_size(self,):
    return len(self.vocab_oplabel)

  def get_feat_vocab_size(self,):
    return len(self.vocab_feats)

  def vocabs_input_mode_toggle(self,fill_vocab=False):
    self.vocab_oplabel._add = fill_vocab
    self.vocab_feat_bundles._add = fill_vocab
    self.vocab_feats._add = fill_vocab


  def read_action_file(self,tbname,split,fill_vocab=False):  
    self.vocabs_input_mode_toggle(fill_vocab)

    lang = tbname[:2]
    lid = lang + "_"
    filename = ACTION_FILE_TEMPLATE % (tbname,split)
    sample = {
      "forms" : [],
      "lemmas": [],
      "bundle_feat": [],
      "fine_feat": [],
      "actions":[],
    }
    data = []
    for line in open(filename,'r'):
      line = line.strip('\n')
      if line=='':
        data.append(sample)
        sample = {
          "forms" : [],
          "lemmas": [],
          "bundle_feat": [],
          "fine_feat": [],
          "actions":[],
        }   
        continue
      comps = line.split("\t")
      w,lem,feats = comps[:3]
      op_ids = [self.vocab_oplabel.add(lid + op) for op in comps[3:]]
      w = lid + w
      lem = lid + lem
      sos = self.vocab_feats.get_label_id(SOS)
      eos = self.vocab_feats.get_label_id(EOS)
      
      sample["actions"].append(op_ids)
      sample["lemmas"].append(self.vocab_lemmas.add(lem))
      sample["forms"].append(self.vocab_forms.add(w))
      sample["bundle_feat"].append(self.vocab_feat_bundles.add(feats))
      sample["fine_feat"].append([sos] + [self.vocab_feats.add(feat) for feat in feats.split(";")] + [eos])
      #
    self.vocabs_input_mode_toggle(False)
    return data


  def dump_tokenized(self,data,filename):
    with open(filename,'w') as outfile:
      for sample in data:
        outfile.write(json.dumps(sample))


  def load_vocab(self):
    def read_dump_tok(tb_list,fill_train):
      for tb in tb_list:
        train_tb = self.read_action_file(tb,"train",fill_vocab=fill_train)
        dev_tb = self.read_action_file(tb,"dev",fill_vocab=False)
        test_tb = self.read_action_file(tb,"test",fill_vocab=False)
        for data,split in zip([train_tb,dev_tb,test_tb],["train","dev","test"]):
          self.dump_tokenized(train_tb,TOKENIZED_JSON_TEMPLATE % (tb,split))


    train_tb = self.read_action_file(tb,"train",fill_vocab=True)
    dev_tb = self.read_action_file(tb,"dev",fill_vocab=False)
    test_tb = self.read_action_file(tb,"test",fill_vocab=False)

  def load_data(self,fill_vocab=False):

    
    
    