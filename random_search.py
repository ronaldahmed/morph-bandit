import hyperopt
from hyperopt import hp, fmin, tpe, space_eval

def run_train(args):



def random_search(args):
  loader = DataLoaderAnalizer(args)
  train = loader.load_data("train")
  dev   = loader.load_data("dev")

  train_batch = BatchSegm(train,args.batch_size,args.gpu)
  dev_batch   = BatchSegm(dev,args.batch_size,args.gpu)
  n_vocab = loader.get_vocab_size()

  space = {
    'lr': hp.loguniform('lr', -9, -2),
    'dropout': hp.uniform('dropout', 0, 0.5),
    'emb_size': hp.quniform('mlp_hid_dim', low=50, high=300,q=20),
    'mlp_size': hp.quniform('mlp_hid_dim', low=100, high=300,q=10),
    'batch_size': hp.quniform('batch_size', low=10, high=128,q=10)
  }
  

  # init trainer
  model = Analizer(args,n_vocab)
  trainer = Trainer(model,n_vocab,args)

  # init local vars
  best_dev_loss = 100000000
  best_dev_loss_index = -1
  best_dev_acc = -1
  start_time = monotonic()

  for ep in range(args.epochs):
    train_loss = 0
    i = 0
    for sents,gold in train_batch.get_batch():
      loss = torch.sum(trainer.train_batch(sents, gold, debug=False))
      train_loss += loss
      
      if i % debug_print == (debug_print - 1):
        trainer.update_summary(train_log_step_cnt,train_loss=loss)
        print(".", end="", flush=True)
      i += 1
      train_log_step_cnt += 1

      # if i>10: break
    #
    dev_loss = 0.0
    i = 0
    for sents,gold in dev_batch.get_batch(shuffle=False):
      dev_loss += torch.sum(trainer.eval_batch(sents,gold,debug=False))
      if i % debug_print == (debug_print - 1):
          print(".", end="", flush=True)
      i += 1




if __name__ == '__main__':
  args = analizer_args()
