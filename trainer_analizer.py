from trainer_analizer_bundle import TrainerAnalizerBundle
from trainer_analizer_seq import TrainerAnalizerSeq

class TrainerAnalizer:
  def __init__(self,anlz_model,num_classes,args):
    self.handler = ""
    if   args.tagger_mode == "bundle":
      self.handler = TrainerAnalizerBundle(anlz_model,num_classes,args)
    elif args.tagger_mode == "fine-seq":
      self.handler = TrainerAnalizerSeq(anlz_model,num_classes,args)
    

  @property
  def scheduler(self):
    return self.handler.scheduler

  def train_batch(self, bundle, debug=0):
    return self.handler.train_batch(bundle, debug=debug)

  def eval_batch(self,bundle,debug=0):
    return self.handler.eval_batch(bundle,debug=debug)

  def eval_metrics_batch(self,**kwargs):
    return self.handler.eval_metrics_batch(**kwargs)

  def save_model(self,ep):
    return self.handler.save_model(ep)

  def update_summary(self,**kwargs):
    return self.handler.update_summary(**kwargs)