from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter



def general_arg_parser():
    """ CLI args related to training and testing models. """
    p = ArgumentParser(add_help=False)
    p.add_argument("--seed", help="Random seed", type=int, default=42)
    p.add_argument("--mode", help="Running mode [train,test]", type=str, default="train")
    p.add_argument("--train_file","-train", help="Training --op extended-- file", type=str, required=True)
    p.add_argument("--dev_file"  ,"-dev"  , help="Development --op extended-- file", type=str, required=True)
    p.add_argument("--test_file" ,"-test" , help="Testing --op extended-- file", type=str)
    p.add_argument("--test_hidden" , help="Flag if test set is hidden (no gold label added)",  action='store_true')

    p.add_argument("--epochs", help="Number of epochs", type=int, default=10)
    p.add_argument("--batch_size", help="Batch size", type=int, default=10)
    p.add_argument("--patience", help="Patience parameter (for early stopping)", type=int, default=30)
    p.add_argument("-lr", "--learning_rate", help="Adam Learning rate", type=float, default=1e-3)
    p.add_argument("--clip", help="Gradient clipping", type=float, default=None)
    
    p.add_argument("--gpu", help="Use GPU", action='store_true')
    p.add_argument("--embedding_file", help="Pretrained op-token embedding file", type=str, default=None)
    p.add_argument("--pretrained_form_emb", help="Pretrained word forms embedding file", type=str, default=None)
    p.add_argument("--input_model", help="Model name to load")
    p.add_argument("--model_save_dir", help="where to save the trained models and logs", type=str)

    return p


def morph_analizer_arg_parser():
    """ CLI args related to training models. """
    p = ArgumentParser(add_help=False)
    p.add_argument("--in_mode", help="Input op token mode [coarse,grain]", type=str, default="coarse")
    p.add_argument("--out_mode", help="Output feat label mode [coarse,grain]", type=str, default="coarse")
    p.add_argument("-r", "--scheduler", help="Use reduce learning rate on plateau schedule", action='store_true')
    p.add_argument("-w", "--word_dropout", help="Use word dropout", type=float, default=0)
    p.add_argument("--dropout", help="Use dropout", type=float, default=0)
    p.add_argument("--emb_size", help="Input embeddings size", type=int, default=100)
    p.add_argument("--rnn_size", help="Input embeddings size", type=int, default=100)
    p.add_argument("--rnn_type", help="Type of rnn cell [LSTM,GRU]", type=str, default="LSTM")
    p.add_argument("--mlp_size", help="Input embeddings size", type=int, default=100)
    p.add_argument("--debug", help="Debug", type=int, default=0)
    
    return p



def sopamlp_arg_parser():
    """ CLI args related to the MLP module """
    p = ArgumentParser(add_help=False)
    p.add_argument("-d", "--mlp_hidden_dim", help="MLP hidden dimension", type=int, default=25)
    p.add_argument("-y", "--num_mlp_layers", help="Number of MLP layers", type=int, default=2)
    return p


def soft_pattern_arg_parser():
    """ CLI args related to SoftPatternsClassifier """
    from utils import SHARED_SL_PARAM_PER_STATE_PER_PATTERN, \
                      SHARED_SL_SINGLE_PARAM
    p = ArgumentParser(add_help=False)
    p.add_argument("-u", "--use_rnn", help="Use an RNN underneath soft-patterns", action="store_true")
    p.add_argument("-p", "--patterns",
                   help="Pattern lengths and numbers: an underscore separated list of length-number pairs",
                   default="5-50_4-50_3-50_2-50")
    p.add_argument("--pre_computed_patterns", help="File containing pre-computed patterns")
    p.add_argument("--maxplus",
                   help="Use max-plus semiring instead of plus-times",
                   default=False, action='store_true')
    p.add_argument("--maxtimes",
                   help="Use max-times semiring instead of plus-times",
                   default=False, action='store_true')
    p.add_argument("--bias_scale_param",
                   help="Scale bias term by this parameter",
                   default=0.1, type=float)
    p.add_argument("--eps_scale",
                   help="Scale epsilon by this parameter",
                   default=None, type=float)
    p.add_argument("--self_loop_scale",
                   help="Scale self_loop by this parameter",
                   default=None, type=float)
    p.add_argument("--no_eps", help="Don't use epsilon transitions", action='store_true')
    p.add_argument("--no_sl", help="Don't use self loops", action='store_true')
    p.add_argument("--shared_sl",
                   help="Share main path and self loop parameters, where self loops are discounted by a self_loop_parameter. "+
                           str(SHARED_SL_PARAM_PER_STATE_PER_PATTERN)+
                           ": one parameter per state per pattern, "+str(SHARED_SL_SINGLE_PARAM)+
                           ": a global parameter.", type=int, default=0)
    # MLP sopa layer
    p.add_argument("-d", "--mlp_hidden_dim", help="MLP hidden dimension", type=int, default=25)
    p.add_argument("-y", "--num_mlp_layers", help="Number of MLP layers", type=int, default=2)
    # LSTM sopa module
    p.add_argument("--hidden_dim", help="RNN hidden dimension", type=int, default=100)

    return p





def analizer_args():
  parser = ArgumentParser(description=__doc__,
                          formatter_class=ArgumentDefaultsHelpFormatter,
                          parents=[soft_pattern_arg_parser(), morph_analizer_arg_parser(), general_arg_parser()])
  return parser.parse_args()