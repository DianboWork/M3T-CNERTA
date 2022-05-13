import argparse, os, torch
import random
import numpy as np
import pickle
from utils.data import Data
from module.singlemodal_tagging_model import SinglemodalTaggingModel


def str2bool(v):
    return v.lower() in ('true')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    return arg


def get_args():
    args, unparsed = parser.parse_known_args()
    if len(unparsed) > 1:
        print("Unparsed args: {}".format(unparsed))
    return args, unparsed


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    data_arg = add_argument_group('Data')
    data_arg.add_argument('--dataset_name', type=str, default="Aishell-NER")
    data_arg.add_argument('--train_file', type=str, default="./data/new_text/train.json")
    data_arg.add_argument('--valid_file', type=str, default="./data/new_text/new_valid.json")
    data_arg.add_argument('--test_file', type=str, default="./data/new_text/new_test.json")
    data_arg.add_argument('--generated_data_directory', type=str, default="./data/generated_data/")
    data_arg.add_argument('--generated_param_directory', type=str, default="./data/generated_data/model_param/")
    data_arg.add_argument('--bert_directory', type=str, default="./plm/bert_base_chinese/")
    data_arg.add_argument('--macbert_directory', type=str, default="./plm/macbert/")
    data_arg.add_argument('--wobert_directory', type=str, default="./plm/wobert/")
    data_arg.add_argument('--zen_directory', type=str, default="./plm/zen/")
    data_arg.add_argument('--emb_file', type=str, default="/home/suidianbo/data/ch_embeddings/gigaword_chn.char50d.vec")

    data_arg.add_argument('--max_seq_length', type=int, default=50)
    data_arg.add_argument('--schema', type=str, default="BIO", choices=["BIO", "BILOU"])
    data_arg.add_argument('--ner_type', type=str, default="Flat_NER", choices=['Nested_NER', 'Flat_NER'])
    data_arg.add_argument('--use_audio_feature', type=str2bool, default=False)
    data_arg.add_argument('--vocab_path', type=str, default="./data/vocab.json")


    model_arg = add_argument_group('Model')
    model_arg.add_argument('--text_encoder', type=str, default="BERT", choices=['BERT', 'ZEN', 'MacBERT', 'WoBERT'])
    model_arg.add_argument('--use_emb', type=str2bool, default=True)
    model_arg.add_argument('--emb_dim', type=int, default=256)
    model_arg.add_argument('--random_emb', type=str2bool, default=True)

    learning_arg = add_argument_group('Learning')
    learning_arg.add_argument('--fix_embeddings', type=str2bool, default=False)
    learning_arg.add_argument('--crf_reduction', type=str, default="mean", choices=["sum", "mean"])
    learning_arg.add_argument('--dropout', type=float, default=0.1)
    learning_arg.add_argument('--batch_size', type=int, default=16)
    learning_arg.add_argument('--max_epoch', type=int, default=10)
    learning_arg.add_argument('--gradient_accumulation_steps', type=int, default=1)
    learning_arg.add_argument('--textual_encoder_lr', type=float, default=1e-5)
    learning_arg.add_argument('--crf_lr', type=float, default=0.1)
    learning_arg.add_argument('--lr_decay', type=float, default=0.01)
    learning_arg.add_argument('--weight_decay', type=float, default=1e-5)
    learning_arg.add_argument('--max_grad_norm', type=float, default=0)
    learning_arg.add_argument('--optimizer', type=str, default='AdamW', choices=['Adam', 'AdamW', 'RAdam', "SGD", 'RAdamW', 'AdaBelief'])
    learning_arg.add_argument('--adversarial_training', type=str2bool, default=False)

    misc_arg = add_argument_group('MISC')
    misc_arg.add_argument('--refresh', type=str2bool, default=True)
    misc_arg.add_argument('--use_gpu', type=str2bool, default=True)
    misc_arg.add_argument('--visible_gpu', type=int, default=2)
    misc_arg.add_argument('--random_seed', type=int, default=10)

    args, unparsed = get_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.visible_gpu)
    for arg in vars(args):
        print(arg, ":",  getattr(args, arg))
    set_seed(args.random_seed)
    # data = Data(args)
    with open(args.generated_data_directory + "unified_data_full.pkl", "rb") as f:
        data = pickle.load(f)
    # with open(args.generated_data_directory+"singlemodal.pkl", "wb") as f:
    #     pickle.dump(data, f)
    # with open(args.generated_data_directory+"singlemodal.pkl", 'rb') as f:
    #     data = pickle.load(f)
    data.show_data_summary()
    model = SinglemodalTaggingModel(args, data)
    if args.use_gpu:
        model = model.cuda()
    model.load_state_dict(torch.load("/home/suidianbo/program/unified_encoder/data/generated_data/model_param/pure_BERT_Flat_NER_epoch_5_f1_0.7666.model"))
    # for n, p in model.named_parameters():
    #     print(n)
    # # # model = SetPred4NNER(args, data.relational_alphabet.size())
    if args.adversarial_training:
        from trainer.fgm_trainer import Trainer
    else:
        from trainer.trainer import Trainer
    trainer = Trainer(model, data, args)

    trainer.output("valid", "./bert_flat_7666_dev.txt")