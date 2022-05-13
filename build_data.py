import argparse, os, torch
import random
import numpy as np
import json
import pickle
from utils.data import Data


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
    data_arg.add_argument('--train_file', type=str, default="./data/new_text/train_5.json")
    data_arg.add_argument('--valid_file', type=str, default="./data/new_text/new_valid.json")
    data_arg.add_argument('--test_file', type=str, default="./data/new_text/new_test.json")
    data_arg.add_argument('--generated_data_directory', type=str, default="./data/generated_data/")
    data_arg.add_argument('--generated_param_directory', type=str, default="./data/generated_data/model_param/")
    data_arg.add_argument('--bert_directory', type=str, default="./plm/bert_base_chinese/")
    data_arg.add_argument('--macbert_directory', type=str, default="./plm/macbert/")
    data_arg.add_argument('--wobert_directory', type=str, default="./plm/wobert/")
    data_arg.add_argument('--zen_directory', type=str, default="./plm/zen/")
    data_arg.add_argument('--emb_file', type=str, default="/home/suidianbo/data/ch_embeddings/gigaword_chn.char50d.vec")
    data_arg.add_argument('--vocab_path', type=str, default="./data/vocab.json")

    data_arg.add_argument('--max_seq_length', type=int, default=50)
    data_arg.add_argument('--schema', type=str, default="BIO", choices=["BIO", "BILOU"])
    data_arg.add_argument('--ner_type', type=str, default="Flat_NER", choices=['Nested_NER', 'Flat_NER'])
    data_arg.add_argument('--use_audio_feature', type=str2bool, default=True)
    data_arg.add_argument('--audio_directory', type=str, default="/home/suidianbo/aishell_ner_audio/")


    feature_arg = add_argument_group('Audio-Feature')
    feature_arg.add_argument('--normalization', type=str2bool, default=True)
    feature_arg.add_argument('--apply_spec_augment', type=str2bool, default=True)
    feature_arg.add_argument('--gaussian_noise', type=float, default=0)
    feature_arg.add_argument('--num_mel_bins', type=int, default=40, choices=[40, 80])


    model_arg = add_argument_group('Model')
    model_arg.add_argument('--text_encoder', type=str, default="ZEN", choices=['BERT', 'ZEN', 'MacBERT', 'WoBERT'])

    misc_arg = add_argument_group('MISC')
    misc_arg.add_argument('--use_gpu', type=str2bool, default=True)
    misc_arg.add_argument('--random_seed', type=int, default=10)

    args, unparsed = get_args()

    for arg in vars(args):
        print(arg, ":",  getattr(args, arg))
    set_seed(args.random_seed)
    # data = Data(args)
    # with open(args.generated_data_directory+"unified_data_5.pkl", "wb") as f:
    #     pickle.dump(data, f)
    with open(args.generated_data_directory + "unified_data_full.pkl", "rb") as f:
        data = pickle.load(f)

