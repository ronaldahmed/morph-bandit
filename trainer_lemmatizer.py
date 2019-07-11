from trainer_lemmatizer_mle import TrainerLemmatizerMLE
from trainer_lemmatizer_mrt import TrainerLemmatizerMRT


def TrainerLemmatizer(lem_model,loader,args):
  if   args.lem_loss == "mle":
    return TrainerLemmatizerMLE(lem_model,loader,args)

  elif args.lem_loss == "mrt":
    return TrainerLemmatizerMRT(lem_model,loader,args)
  