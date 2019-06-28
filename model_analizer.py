from model_analizer_bundle import AnalizerBundle
from model_analizer_seq import AnalizerSeq


def analizer(args,nvocab):
  if   args.tagger_mode == 'bundle':
    return AnalizerBundle(args,nvocab)
  elif args.tagger_mode == 'fine-seq':
    return AnalizerSeq(args,nvocab)
